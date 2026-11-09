"""M3: planner.

The planner is the only part of the agent that decides *what* to do. It turns a
world brief into a small list of high-level steps, then hands those steps to the
deterministic executor (M2). It has no power of its own: every step is validated
against the live snapshot and against a safety policy before anything is sent to
the game.

Two planners are provided:

* ``LLMPlanner`` asks an OpenRouter model for a JSON plan.
* ``HeuristicPlanner`` is a small offline fallback (nearest free ore → drill,
  a Duo nearby, or scout toward unexplored ground) used when no API key is set.

Safety is enforced in ``validate_plan``, never in the model prompt alone:

* the planner may not emit ``break`` or ``pause`` steps;
* it may not build within ``safe_margin`` tiles of any existing own building;
* it may not build on a core, on an occupied tile, or outside the map;
* a mechanical drill must sit on known ore, and other blocks must not;
* plans are capped in length and build distance from the ship.

That means a hallucinating model can waste a little copper, but it cannot delete
the base the user built.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .executor import Executor
from .memory import WorldMemory
from .rpc import Bridge

log = logging.getLogger("ai_ship.planner")

TILE = 8
DEFAULT_STATE = Path(__file__).resolve().parent.parent / "state" / "world-memory.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
# Free OpenRouter models share upstream pools and are often rate-limited, so
# try a few different providers before giving up and using the heuristic.
DEFAULT_FALLBACK_MODELS = ",".join([
    "nvidia/nemotron-3.5-super-vl:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
    "liquid/lfm-3-1.7b:free",
    "recursive/flint-v1.0:free",
])

# Mindustry block rotation, verified live with Building.front():
#   0 = north (-y), 1 = south (+y), 2 = west (-x), 3 = east (+x)
ROTATION_NAMES = {0: "north", 1: "south", 2: "west", 3: "east"}
DIRECTION_ROT = {"north": 0, "south": 1, "west": 2, "east": 3,
                 "up": 0, "down": 1, "left": 2, "right": 3,
                 "n": 0, "s": 1, "w": 2, "e": 3}

BUILDABLE = frozenset({
    "conveyor",
    "mechanical-drill",
    "duo",
    "router",
    "distributor",
    "wall",
    "scatter",
    "hail",
})

BLOCK_ALIASES = {
    "drill": "mechanical-drill",
    "mechanical drill": "mechanical-drill",
    "mechanical_drill": "mechanical-drill",
    "belt": "conveyor",
    "conveyor belt": "conveyor",
    "conveyor_belt": "conveyor",
    "turret": "duo",
    "duo turret": "duo",
    "distributor": "distributor",
    "router": "router",
    "wall": "wall",
    "scatter": "scatter",
    "hail": "hail",
}


class PlannerError(RuntimeError):
    """The LLM call or its output could not be turned into a plan."""


class PlanValidationError(PlannerError):
    """The model produced steps that are unsafe or impossible."""


@dataclass
class Policy:
    """Hard limits the planner cannot talk its way out of."""

    allowed_blocks: frozenset[str] = BUILDABLE
    safe_margin: int = 2
    max_steps: int = 20
    max_build_distance: float = 800.0
    allow_break: bool = False
    allow_pause: bool = False


@dataclass
class Plan:
    steps: list[dict[str, Any]]
    rationale: str = ""
    source: str = "llm"
    model: str = ""
    raw: str = ""


# --------------------------------------------------------------------- transport


class ChatTransport(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class OpenRouterTransport:
    """Minimal OpenRouter client using only the standard library.

    ``urllib`` is blocking, so the actual request runs in a worker thread; the
    event loop stays free to keep the bridge heartbeating.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 90.0,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        json_mode: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self.api_key:
            raise PlannerError(
                "OPENROUTER_API_KEY is not set; use --heuristic or set the key"
            )

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("json_mode", self.json_mode):
            payload["response_format"] = {"type": "json_object"}
            # only route to providers that actually honour response_format;
            # otherwise some providers return free-form reasoning instead
            payload["provider"] = {"require_parameters": True}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pranavvatlam-sudo/mindustry-ai-ship",
            "X-Title": "AI Ship",
        }
        return await asyncio.to_thread(self._post, body, headers)

    def _post(self, body: bytes, headers: dict[str, str]) -> str:
        request = urllib.request.Request(
            OPENROUTER_URL, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
            except Exception:
                pass
            raise PlannerError(
                f"OpenRouter HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PlannerError(f"OpenRouter unreachable: {exc.reason}") from exc
        except (OSError, ValueError) as exc:
            raise PlannerError(f"OpenRouter request failed: {exc}") from exc

        return extract_content(data)


def extract_content(payload: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenRouter response."""
    choices = payload.get("choices") or []
    if not choices:
        error = payload.get("error")
        if error:
            raise PlannerError(f"OpenRouter error: {error}")
        raise PlannerError("OpenRouter returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # some providers return content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not content:
        # reasoning models sometimes leave content empty and put the answer in
        # a separate reasoning field
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            content = reasoning
    if not content:
        raise PlannerError(
            "OpenRouter returned an empty message "
            f"(keys: {', '.join(sorted(message)) or 'none'})"
        )
    return str(content)


_RETRY_MARKERS = (
    "429",
    "rate-limited",
    "temporarily rate",
    "502",
    "503",
    "504",
    "unreachable",
    "timed out",
    "empty message",
)


class ModelFallbackTransport:
    """Try a list of models in order when a provider is rate-limited.

    OpenRouter's free models share upstream pools and are frequently 429'd.
    Failed 429/5xx attempts do not count against the free daily quota, so
    walking a short fallback list is cheap.
    """

    def __init__(self, candidates: list[tuple[str, ChatTransport]]) -> None:
        if not candidates:
            raise ValueError("at least one model candidate is required")
        self.candidates = candidates
        self.last_model = ""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        failures: list[str] = []
        call_kwargs = {key: value for key, value in kwargs.items() if key != "model"}
        for model, transport in self.candidates:
            try:
                text = await transport.complete(messages, model=model, **call_kwargs)
            except PlannerError as exc:
                message = str(exc)
                if any(marker in message.lower() for marker in _RETRY_MARKERS):
                    failures.append(f"{model}: {message.splitlines()[0][:160]}")
                    log.warning("model %s unavailable, trying the next one", model)
                    continue
                raise
            self.last_model = model
            return text
        raise PlannerError("all planner models failed: " + " | ".join(failures))


# ------------------------------------------------------------------------ parsing


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _looks_like_plan(data: Any) -> bool:
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        return True
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(isinstance(step, dict) and "op" in step for step in data)
    )


def _plan_from_data(data: Any) -> tuple[list[dict[str, Any]], str]:
    rationale = ""
    if isinstance(data, dict):
        rationale = str(data.get("rationale") or data.get("reason") or "")
        data = data.get("steps")
    if not isinstance(data, list):
        raise PlannerError("planner JSON has no steps array")
    return data, rationale


def parse_plan_json(text: str) -> tuple[list[dict[str, Any]], str]:
    """Return ``(steps, rationale)`` from the model's text.

    Models sometimes wrap the JSON in prose or think out loud first. We try the
    whole string, then scan for balanced JSON values and pick the one that
    actually looks like a plan (an object with ``steps`` or an array of ops).
    """
    cleaned = text.strip()
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        data = json.loads(cleaned)
        if _looks_like_plan(data):
            return _plan_from_data(data)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, Any]] = []
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            data, end = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            continue
        if _looks_like_plan(data):
            candidates.append((end - index, data))

    if not candidates:
        raise PlannerError(f"no plan JSON in planner reply: {text[:200]!r}")

    def score(item: tuple[int, Any]) -> tuple[int, int]:
        length, data = item
        # prefer an explicit {"steps": [...]} object over a bare array
        explicit = 1 if isinstance(data, dict) and "steps" in data else 0
        return (explicit, length)

    _, best = max(candidates, key=score)
    return _plan_from_data(best)


# ---------------------------------------------------------------------- validation


def _norm_block(name: Any) -> str:
    raw = str(name or "").strip().lower().replace("_", " ")
    raw = BLOCK_ALIASES.get(raw, raw)
    return raw.replace(" ", "-")


def _tile_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a tile coordinate")
    return int(value)


def _rot_int(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in DIRECTION_ROT:
            return DIRECTION_ROT[value]
    if isinstance(value, bool):
        raise ValueError("bool is not a rotation")
    rot = int(value)
    if not 0 <= rot <= 3:
        raise ValueError(f"rotation {rot} is not 0..3")
    return rot


def protected_tiles(memory: WorldMemory) -> set[tuple[int, int]]:
    """Tiles the planner must keep clear: every own building and every core."""
    team = memory.team
    tiles: set[tuple[int, int]] = set()
    for block in memory.blocks.values():
        if team is None or block.team == team:
            tiles.add((block.x, block.y))
    for core in memory.cores:
        try:
            tiles.add((int(core["tx"]), int(core["ty"])))
        except (KeyError, TypeError, ValueError):
            continue
    return tiles


def _too_close(
    x: int, y: int, protected: set[tuple[int, int]], margin: int
) -> tuple[int, int] | None:
    for dx in range(-margin, margin + 1):
        for dy in range(-margin, margin + 1):
            if (x + dx, y + dy) in protected:
                return (x + dx, y + dy)
    return None


def validate_plan(
    steps: Iterable[dict[str, Any]],
    memory: WorldMemory,
    policy: Policy | None = None,
    ship: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize and validate a plan, or raise ``PlanValidationError``."""
    policy = policy or Policy()
    raw_steps = list(steps)
    if not raw_steps:
        raise PlanValidationError("plan is empty")
    if len(raw_steps) > policy.max_steps:
        raise PlanValidationError(
            f"plan has {len(raw_steps)} steps; the limit is {policy.max_steps}"
        )

    ship = ship if ship is not None else (memory.ship or {})
    ship_x = float(ship.get("x") or 0.0)
    ship_y = float(ship.get("y") or 0.0)
    ship_alive = bool(ship.get("alive"))

    protected = protected_tiles(memory)
    used_tiles: set[tuple[int, int]] = set()
    clean: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            errors.append(f"step {index}: not an object")
            continue
        op = str(raw.get("op") or "").strip().lower()
        where = f"step {index} ({op or '?'})"

        if op == "build":
            if not ship_alive:
                errors.append(f"{where}: ship is not alive")
                continue
            block = _norm_block(raw.get("block"))
            if block not in policy.allowed_blocks:
                errors.append(f"{where}: block {block!r} is not allowed")
                continue
            try:
                tx = _tile_int(raw.get("tx"))
                ty = _tile_int(raw.get("ty"))
                rot = _rot_int(raw.get("rot", 0))
            except (TypeError, ValueError) as exc:
                errors.append(f"{where}: bad tile/rotation: {exc}")
                continue

            if memory.map_w and not (0 <= tx < memory.map_w and 0 <= ty < memory.map_h):
                errors.append(f"{where}: tile ({tx},{ty}) is outside the map")
                continue
            if (tx, ty) in used_tiles:
                errors.append(f"{where}: tile ({tx},{ty}) is used twice")
                continue
            if (tx, ty) in protected:
                errors.append(f"{where}: tile ({tx},{ty}) holds an own building or core")
                continue
            near = _too_close(tx, ty, protected, policy.safe_margin)
            if near is not None:
                errors.append(
                    f"{where}: too close to own building at {near} "
                    f"(margin {policy.safe_margin})"
                )
                continue
            if (tx, ty) in memory.blocks:
                errors.append(
                    f"{where}: tile ({tx},{ty}) already has "
                    f"{memory.blocks[(tx, ty)].block}"
                )
                continue

            ore = memory.ores.get((tx, ty))
            if block == "mechanical-drill" and ore is None:
                errors.append(f"{where}: mechanical drill must be on known ore")
                continue
            if block != "mechanical-drill" and ore is not None:
                errors.append(
                    f"{where}: would cover {ore.item} ore at ({tx},{ty}); "
                    "save ore for a drill"
                )
                continue

            px, py = tx * TILE + TILE / 2, ty * TILE + TILE / 2
            if math.hypot(px - ship_x, py - ship_y) > policy.max_build_distance:
                errors.append(
                    f"{where}: target is farther than "
                    f"{policy.max_build_distance:.0f}px from the ship"
                )
                continue

            used_tiles.add((tx, ty))
            clean.append({"op": "build", "block": block,
                          "tx": tx, "ty": ty, "rot": rot})

        elif op == "move":
            try:
                x = float(raw.get("x"))
                y = float(raw.get("y"))
            except (TypeError, ValueError):
                errors.append(f"{where}: move needs numeric x and y")
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                errors.append(f"{where}: move target is not finite")
                continue
            if memory.map_w and not (0 <= x < memory.map_w * TILE
                                     and 0 <= y < memory.map_h * TILE):
                errors.append(f"{where}: move target ({x:.0f},{y:.0f}) is outside the map")
                continue
            try:
                timeout = float(raw.get("timeout", 120.0))
            except (TypeError, ValueError):
                errors.append(f"{where}: bad move timeout")
                continue
            timeout = max(5.0, min(300.0, timeout))
            clean.append({"op": "move", "x": x, "y": y, "timeout": timeout})

        elif op == "hold":
            clean.append({"op": "hold"})

        elif op == "break":
            if not policy.allow_break:
                errors.append(f"{where}: the planner is not allowed to break blocks")
            else:
                try:
                    tx = _tile_int(raw.get("tx"))
                    ty = _tile_int(raw.get("ty"))
                except (TypeError, ValueError):
                    errors.append(f"{where}: break needs tile coordinates")
                    continue
                if (tx, ty) in protected:
                    errors.append(f"{where}: refusing to break own building/core")
                else:
                    clean.append({"op": "break", "tx": tx, "ty": ty})

        elif op == "pause":
            if not policy.allow_pause:
                errors.append(f"{where}: the planner is not allowed to pause the game")
            else:
                clean.append({"op": "pause", "paused": bool(raw.get("paused"))})

        else:
            errors.append(f"{where}: unknown op {op!r}")

    if errors:
        raise PlanValidationError("\n".join(errors))
    return clean


# -------------------------------------------------------------------------- briefs


def _cluster_bounds(blocks: Iterable[Any]) -> dict[str, Any]:
    xs = [b.x for b in blocks]
    ys = [b.y for b in blocks]
    if not xs:
        return {}
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}


def ore_candidates(
    memory: WorldMemory,
    ship_x: float,
    ship_y: float,
    policy: Policy,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Free ore tiles the policy would accept a drill on, nearest first."""
    protected = protected_tiles(memory)
    rows: list[dict[str, Any]] = []
    for ore in memory.ores.values():
        key = (ore.x, ore.y)
        if key in memory.blocks or key in protected:
            continue
        if _too_close(ore.x, ore.y, protected, policy.safe_margin) is not None:
            continue
        px, py = ore.x * TILE + TILE / 2, ore.y * TILE + TILE / 2
        distance = math.hypot(px - ship_x, py - ship_y)
        if distance > policy.max_build_distance:
            continue
        rows.append({
            "item": ore.item,
            "tx": ore.x,
            "ty": ore.y,
            "x": round(px, 1),
            "y": round(py, 1),
            "distance": round(distance),
        })
    rows.sort(key=lambda row: (row["distance"], row["tx"], row["ty"]))
    return rows[:limit]


def build_spots(
    memory: WorldMemory,
    ship_x: float,
    ship_y: float,
    policy: Policy,
    radius: int = 12,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Empty tiles near the ship where a non-drill block is allowed."""
    protected = protected_tiles(memory)
    tx0 = int(ship_x // TILE)
    ty0 = int(ship_y // TILE)
    rows: list[dict[str, Any]] = []
    for tx in range(tx0 - radius, tx0 + radius + 1):
        for ty in range(ty0 - radius, ty0 + radius + 1):
            if memory.map_w and not (0 <= tx < memory.map_w and 0 <= ty < memory.map_h):
                continue
            if (tx, ty) in memory.blocks or (tx, ty) in memory.ores or (tx, ty) in protected:
                continue
            if _too_close(tx, ty, protected, policy.safe_margin) is not None:
                continue
            px, py = tx * TILE + TILE / 2, ty * TILE + TILE / 2
            distance = math.hypot(px - ship_x, py - ship_y)
            if distance > policy.max_build_distance:
                continue
            rows.append({
                "tx": tx,
                "ty": ty,
                "x": round(px, 1),
                "y": round(py, 1),
                "distance": round(distance),
            })
    rows.sort(key=lambda row: (row["distance"], row["tx"], row["ty"]))
    return rows[:limit]


def build_brief(
    memory: WorldMemory,
    goal: str,
    policy: Policy | None = None,
) -> dict[str, Any]:
    """A compact, model-friendly description of the world."""
    policy = policy or Policy()
    ship = memory.ship or {}
    ship_x = float(ship.get("x") or 0.0)
    ship_y = float(ship.get("y") or 0.0)
    team = memory.team

    own = [b for b in memory.blocks.values() if team is None or b.team == team]
    counts = Counter(b.block for b in own)

    core = None
    if memory.cores:
        core = memory.cores[0]
        core = {
            "tx": core.get("tx"),
            "ty": core.get("ty"),
            "x": core.get("x"),
            "y": core.get("y"),
            "items": core.get("items") or {},
            "health": core.get("health"),
            "maxHealth": core.get("maxHealth"),
        }

    enemies = [
        {
            "type": e.type,
            "team": e.team,
            "x": round(e.x),
            "y": round(e.y),
            "health": round(e.health),
            "age": round(memory.age(e)),
        }
        for e in memory.enemies_recent(max_age=60.0)
    ][:8]

    frontier = []
    if memory.ship:
        for tx, ty in memory.frontier(limit=64):
            px, py = tx * TILE + TILE / 2, ty * TILE + TILE / 2
            frontier.append({"x": px, "y": py,
                             "distance": round(math.hypot(px - ship_x, py - ship_y))})
        frontier.sort(key=lambda row: row["distance"])
        frontier = frontier[:8]

    return {
        "goal": goal,
        "map": {"w": memory.map_w, "h": memory.map_h, "tile": TILE},
        "wave": memory.wave,
        "wavetime": None if memory.wavetime is None else round(memory.wavetime),
        "winWave": memory.win_wave,
        "team": team,
        "ship": {
            "alive": bool(ship.get("alive")),
            "x": round(ship_x, 1),
            "y": round(ship_y, 1),
            "tx": int(ship_x // TILE),
            "ty": int(ship_y // TILE),
            "health": ship.get("health"),
            "maxHealth": ship.get("maxHealth"),
            "range": ship.get("range"),
            "vision": ship.get("vision"),
        },
        "core": core,
        "own_buildings": {
            "total": len(own),
            "counts": dict(sorted(counts.items())),
            "bounds": _cluster_bounds(own),
        },
        "rotation": {"0": "north", "1": "south", "2": "west", "3": "east"},
        "free_ore": ore_candidates(memory, ship_x, ship_y, policy),
        "build_spots": build_spots(memory, ship_x, ship_y, policy),
        "enemies": enemies,
        "frontier": frontier,
    }


SYSTEM_PROMPT = """\
You are the planner for one AI ship in Mindustry v160, sector Ground Zero.
You do not control the game directly. You output a short JSON plan that a
deterministic executor will validate and carry out.

Reply with ONE JSON object and nothing else:
{"rationale": "one short sentence", "steps": [ ... ]}

Allowed steps:
  {"op":"build","block":"mechanical-drill|conveyor|duo|router|distributor|wall|scatter|hail","tx":<tile x>,"ty":<tile y>,"rot":<0-3>}
  {"op":"move","x":<world px>,"y":<world px>}
  {"op":"hold"}

Rotation: 0=north (-y), 1=south (+y), 2=west (-x), 3=east (+x).
The map tile size is 8 px. Quotes in the brief are tile coordinates; ship,
core, ore and frontier entries also include world pixels.

Hard rules (the executor will reject violations):
- Never output "break", "pause", or any other op. You cannot demolish anything.
- Never build within 2 tiles (Chebyshev) of an existing own building or the core.
- A mechanical drill must be placed on a free ore tile listed in free_ore.
- Do not place other blocks on ore tiles; ore is for drills.
- Stay within 800 px of the ship; prefer the nearest candidates.
- At most 20 steps. Prefer 3-8. Keep the plan small and useful.

Gameplay guidance:
- The core is the only place items are delivered. A drill alone fills up and
  stops. To deliver, place a conveyor line from the drill toward the core, with
  every conveyor pointing toward the next tile (rot is the direction it moves
  items). Do not cross ore tiles.
- A Duo needs copper ammo; it is cheap and good early defense.
- Do not modify or rely on the player's existing base beyond reading it.
- If there is no safe build candidate, move toward a frontier point and hold.
"""


def _messages(brief: dict[str, Any], repair: str | None = None) -> list[dict[str, str]]:
    user = "World brief:\n" + json.dumps(brief, indent=1, sort_keys=True)
    if repair:
        user += (
            "\n\nYour previous reply was rejected by the safety validator:\n"
            + repair
            + "\nReturn a corrected JSON plan."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------- planners


class LLMPlanner:
    def __init__(
        self,
        transport: ChatTransport,
        model: str = DEFAULT_MODEL,
        policy: Policy | None = None,
        repair_attempts: int = 1,
    ) -> None:
        self.transport = transport
        self.model = model
        self.policy = policy or Policy()
        self.repair_attempts = repair_attempts

    async def plan(self, goal: str, memory: WorldMemory) -> Plan:
        brief = build_brief(memory, goal, self.policy)
        messages = _messages(brief)
        last_error: Exception | None = None

        for attempt in range(self.repair_attempts + 1):
            text = await self.transport.complete(messages, model=self.model)
            try:
                raw_steps, rationale = parse_plan_json(text)
                steps = validate_plan(raw_steps, memory, self.policy)
                used_model = getattr(self.transport, "last_model", "") or self.model
                return Plan(steps=steps, rationale=rationale, source="llm",
                            model=used_model, raw=text)
            except PlannerError as exc:
                last_error = exc
                log.warning("planner reply rejected (attempt %d): %s", attempt + 1, exc)
                if attempt >= self.repair_attempts:
                    break
                messages = _messages(brief, str(exc))

        assert last_error is not None
        raise last_error


class FallbackPlanner:
    """Try several models in order until one produces a valid plan.

    A model can fail in two different ways: the provider is rate-limited
    (transport error), or the model returns prose instead of a usable plan
    (parse/validation error). Both are retryable on the next model; an auth
    error is not.
    """

    def __init__(
        self,
        candidates: list[tuple[str, ChatTransport]],
        policy: Policy | None = None,
        repair_attempts: int = 1,
    ) -> None:
        if not candidates:
            raise ValueError("at least one model candidate is required")
        self.candidates = candidates
        self.policy = policy or Policy()
        self.repair_attempts = repair_attempts

    async def plan(self, goal: str, memory: WorldMemory) -> Plan:
        failures: list[str] = []
        for model, transport in self.candidates:
            try:
                return await LLMPlanner(
                    transport, model=model, policy=self.policy,
                    repair_attempts=self.repair_attempts,
                ).plan(goal, memory)
            except PlannerError as exc:
                message = str(exc)
                failures.append(f"{model}: {message.splitlines()[0][:180]}")
                if "401" in message or "403" in message:
                    raise
                log.warning("model %s could not produce a plan, trying the next", model)
        raise PlannerError("all planner models failed: " + " | ".join(failures))


class HeuristicPlanner:
    """Offline fallback: one drill, one Duo, or a scout move. No LLM needed."""

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()

    def plan(self, goal: str, memory: WorldMemory) -> Plan:
        ship = memory.ship or {}
        ship_x = float(ship.get("x") or 0.0)
        ship_y = float(ship.get("y") or 0.0)
        steps: list[dict[str, Any]] = []
        notes: list[str] = []

        ores = ore_candidates(memory, ship_x, ship_y, self.policy)
        spots = build_spots(memory, ship_x, ship_y, self.policy)

        if ores:
            ore = ores[0]
            steps.append({"op": "build", "block": "mechanical-drill",
                          "tx": ore["tx"], "ty": ore["ty"], "rot": 0})
            notes.append(f"drill {ore['item']} at ({ore['tx']},{ore['ty']})")

            # a Duo close to the drill, but not on the drill tile
            best = None
            best_d = math.inf
            for spot in spots:
                d = math.hypot(spot["tx"] - ore["tx"], spot["ty"] - ore["ty"])
                if d < best_d and d >= 1:
                    best, best_d = spot, d
            if best is not None and best_d <= 6:
                steps.append({"op": "build", "block": "duo",
                              "tx": best["tx"], "ty": best["ty"], "rot": 0})
                notes.append(f"duo at ({best['tx']},{best['ty']})")
        else:
            goal_lower = goal.lower()
            wants_scout = any(
                word in goal_lower
                for word in ("scout", "explore", "travel", "move to", "recon")
            )
            if spots and not wants_scout:
                # no free ore in range: still improve the position with a cheap
                # Duo on ground that the safety policy already approved
                spot = spots[0]
                steps.append({"op": "build", "block": "duo",
                              "tx": spot["tx"], "ty": spot["ty"], "rot": 0})
                notes.append(f"no free ore; duo at ({spot['tx']},{spot['ty']})")
            else:
                frontier = memory.frontier(limit=64)
                if frontier:
                    # the farthest frontier point is the most useful scout target
                    tx, ty = max(
                        frontier,
                        key=lambda p: (p[0] * TILE - ship_x) ** 2
                        + (p[1] * TILE - ship_y) ** 2,
                    )
                    steps.append({"op": "move",
                                  "x": tx * TILE + TILE / 2,
                                  "y": ty * TILE + TILE / 2})
                    notes.append(f"scout toward ({tx},{ty})")

        steps.append({"op": "hold"})
        try:
            clean = validate_plan(steps, memory, self.policy)
        except PlanValidationError as exc:
            log.warning("heuristic plan invalid (%s); holding instead", exc)
            clean = [{"op": "hold"}]
        return Plan(steps=clean, rationale="; ".join(notes) or "hold position",
                    source="heuristic")


# --------------------------------------------------------------------------- CLI


async def _plan_offline(args: argparse.Namespace) -> int:
    memory = WorldMemory.load(args.state)
    if not memory.blocks and not memory.ores:
        print(f"no memory at {args.state}; run `python -m ai_ship.memory --once` first")
        return 2
    return await _plan_and_maybe_run(args, memory, None)


async def _plan_live(args: argparse.Namespace) -> int:
    memory = WorldMemory.load(args.state)
    bridge = Bridge(host=args.host, port=args.port)
    await bridge.start()
    try:
        session = await bridge.wait_connected(timeout=args.wait)
        print("link up from", session.peer)
        print("hello:", json.dumps(session.hello, sort_keys=True))
        snap = await bridge.snapshot()
        memory.update(snap)
        if snap.get("paused"):
            print("warning: the game is paused; builds will not progress until you focus it")
        memory.save(args.state)
        return await _plan_and_maybe_run(args, memory, bridge)
    except asyncio.TimeoutError:
        print("no connection from the game")
        return 2
    finally:
        await bridge.stop()


async def _plan_and_maybe_run(
    args: argparse.Namespace,
    memory: WorldMemory,
    bridge: Bridge | None,
) -> int:
    policy = Policy(safe_margin=max(1, args.margin),
                    max_steps=max(1, args.max_steps),
                    max_build_distance=args.max_distance)

    print(memory.summary())
    if args.show_brief:
        print(json.dumps(build_brief(memory, args.goal, policy), indent=2))

    if args.heuristic:
        plan = HeuristicPlanner(policy).plan(args.goal, memory)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("OPENROUTER_API_KEY is not set; falling back to --heuristic")
            plan = HeuristicPlanner(policy).plan(args.goal, memory)
        else:
            base = OpenRouterTransport(api_key=api_key, model=args.model,
                                       json_mode=args.json_mode,
                                       max_tokens=args.max_tokens)
            models = [args.model]
            if args.model_fallback:
                for name in args.fallback_models.split(","):
                    name = name.strip()
                    if name and name not in models:
                        models.append(name)
            candidates = [(name, base) for name in models]
            try:
                if len(candidates) > 1:
                    plan = await FallbackPlanner(
                        candidates, policy=policy,
                        repair_attempts=args.repairs,
                    ).plan(args.goal, memory)
                else:
                    plan = await LLMPlanner(
                        base, model=args.model, policy=policy,
                        repair_attempts=args.repairs,
                    ).plan(args.goal, memory)
            except PlannerError as exc:
                print(f"planner failed: {exc}")
                if args.fallback_heuristic:
                    print("falling back to the heuristic planner")
                    plan = HeuristicPlanner(policy).plan(args.goal, memory)
                else:
                    return 1

    print(f"planner: {plan.source}" + (f" ({plan.model})" if plan.model else ""))
    if plan.rationale:
        print("rationale:", plan.rationale)
    print(json.dumps(plan.steps, indent=2))

    if not args.run:
        print("dry run: pass --run --yes to execute these steps")
        return 0
    if not args.yes:
        print("refusing to execute without --yes (this spends real items)")
        return 2
    if bridge is None:
        print("--run requires a live connection; drop --offline")
        return 2

    executor = Executor(bridge, memory)
    results = await executor.plan(plan.steps)
    failed = False
    for index, result in enumerate(results, 1):
        status = "ok" if result.ok else f"FAIL: {result.error}"
        print(f"[{index}] {result.op}: {status}")
        failed = failed or not result.ok

    memory.save(args.state)
    print("memory saved to", args.state)
    return 0 if not failed else 1


async def _main_async(args: argparse.Namespace) -> int:
    if args.offline:
        return await _plan_offline(args)
    return await _plan_live(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M3 planner: turn a goal into validated executor steps"
    )
    parser.add_argument("--goal", default=(
        "Improve the base: place a mechanical drill on a free ore tile and a Duo "
        "turret nearby for defense. Do not touch existing buildings."
    ))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wait", type=float, default=120.0,
                        help="seconds to wait for the mod connection")
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS,
                        help="comma-separated models to try if the primary is rate-limited")
    parser.add_argument("--no-model-fallback", dest="model_fallback",
                        action="store_false", default=True,
                        help="use only --model")
    parser.add_argument("--no-json-mode", dest="json_mode", action="store_false",
                        default=True,
                        help="do not request response_format=json_object "
                             "(for models that reject it)")
    parser.add_argument("--max-tokens", type=int, default=1600,
                        help="max tokens for the planner reply")
    parser.add_argument("--offline", action="store_true",
                        help="plan from saved memory; do not connect to the game")
    parser.add_argument("--heuristic", action="store_true",
                        help="skip the LLM and use the deterministic fallback")
    parser.add_argument("--fallback-heuristic", action="store_true", default=True,
                        help="use the heuristic if the LLM fails (default)")
    parser.add_argument("--no-fallback-heuristic", dest="fallback_heuristic",
                        action="store_false")
    parser.add_argument("--repairs", type=int, default=1,
                        help="how many times to ask the model to fix a rejected plan")
    parser.add_argument("--margin", type=int, default=2,
                        help="tiles to keep clear around existing buildings")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-distance", type=float, default=800.0,
                        help="maximum build distance from the ship, in pixels")
    parser.add_argument("--show-brief", action="store_true")
    parser.add_argument("--run", action="store_true",
                        help="execute the validated plan (requires --yes)")
    parser.add_argument("--yes", action="store_true",
                        help="confirm live execution; spends real items")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
