"""M1: perception memory.

The mod only reports what the ship can see *right now*. This module turns that
stream of snapshots into a map the agent can reason about: ore tiles, buildings
and enemy tracks it has observed, plus the tiles it has explored. Vision stays
honest — anything visible but no longer reported is forgotten (it was mined or
destroyed), while anything outside vision is remembered with its age.

Nothing here talks to the game directly; feed it snapshots from `rpc.Bridge`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .rpc import Bridge, BridgeError

log = logging.getLogger("ai_ship.memory")

TILE = 8
DEFAULT_VISION = 26 * TILE
ENEMY_MERGE_PX = 16.0
ENEMY_TTL = 120.0
STATE_VERSION = 1

# agent/state/world-memory.json, regardless of the working directory
DEFAULT_STATE = Path(__file__).resolve().parent.parent / "state" / "world-memory.json"


@dataclass
class OreTile:
    """An ore tile the ship has seen. ``last_seen`` is wall-clock monotonic."""

    x: int
    y: int
    item: str
    last_seen: float
    sightings: int = 1

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class BlockTile:
    x: int
    y: int
    block: str
    team: int
    rot: int
    last_seen: float
    sightings: int = 1

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class EnemyTrack:
    type: str
    team: int
    x: float
    y: float
    health: float
    last_seen: float
    sightings: int = 1
    player: bool = False

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class MemoryDelta:
    """What changed in one snapshot, so the executor can react to it."""

    reset: bool = False
    ores_added: list[OreTile] = field(default_factory=list)
    ores_removed: list[tuple[int, int]] = field(default_factory=list)
    blocks_added: list[BlockTile] = field(default_factory=list)
    blocks_removed: list[tuple[int, int]] = field(default_factory=list)
    blocks_changed: list[BlockTile] = field(default_factory=list)
    new_tiles: int = 0
    new_enemies: list[EnemyTrack] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.reset
            or self.ores_added
            or self.ores_removed
            or self.blocks_added
            or self.blocks_removed
            or self.blocks_changed
            or self.new_tiles
            or self.new_enemies
        )

    def describe(self) -> str:
        if not self:
            return "no change"
        parts = []
        if self.reset:
            parts.append("RESET")
        if self.ores_added:
            parts.append(f"+{len(self.ores_added)} ore")
        if self.ores_removed:
            parts.append(f"-{len(self.ores_removed)} ore")
        if self.blocks_added:
            parts.append(f"+{len(self.blocks_added)} blocks")
        if self.blocks_removed:
            parts.append(f"-{len(self.blocks_removed)} blocks")
        if self.blocks_changed:
            parts.append(f"~{len(self.blocks_changed)} blocks")
        if self.new_tiles:
            parts.append(f"+{self.new_tiles} tiles")
        if self.new_enemies:
            parts.append(f"+{len(self.new_enemies)} enemies")
        return ", ".join(parts)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _tile_center(tx: int, ty: int) -> tuple[float, float]:
    return (tx * TILE + TILE / 2, ty * TILE + TILE / 2)


class WorldMemory:
    """Accumulated, vision-limited knowledge of the current map."""

    def __init__(self) -> None:
        self.ores: dict[tuple[int, int], OreTile] = {}
        self.blocks: dict[tuple[int, int], BlockTile] = {}
        self.enemies: list[EnemyTrack] = []
        self.seen: set[tuple[int, int]] = set()

        self.map_w = 0
        self.map_h = 0
        self.team: int | None = None
        self.wave: int | None = None
        self.wavetime: float | None = None
        self.win_wave: int | None = None
        self.ship: dict[str, Any] | None = None
        self.cores: list[dict[str, Any]] = []
        self.player: dict[str, Any] | None = None

        self.now = 0.0
        self.game_time = 0.0
        self.snapshots = 0
        self.paused = False

    # ------------------------------------------------------------------ update

    def reset(self) -> None:
        self.ores.clear()
        self.blocks.clear()
        self.enemies.clear()
        self.seen.clear()
        self.map_w = self.map_h = 0
        self.ship = None
        self.cores = []
        self.player = None
        self.wave = self.win_wave = None
        self.wavetime = None
        self.snapshots = 0

    def update(self, snap: dict[str, Any] | None) -> MemoryDelta:
        """Merge one vision-limited snapshot into memory."""
        delta = MemoryDelta()
        if not snap:
            return delta

        self.paused = bool(snap.get("paused"))
        self.game_time = float(snap.get("time") or 0.0)
        if not snap.get("ready"):
            return delta

        world = snap.get("map") or {}
        w = int(world.get("w") or 0)
        h = int(world.get("h") or 0)
        if w > 0 and h > 0 and self.map_w and (w, h) != (self.map_w, self.map_h):
            log.info("map changed %sx%s -> %sx%s, forgetting the old world",
                     self.map_w, self.map_h, w, h)
            self.reset()
            delta.reset = True

        self.map_w, self.map_h = w, h
        self.now = time.monotonic()
        self.snapshots += 1
        self.team = snap.get("team")
        self.wave = snap.get("wave")
        self.wavetime = snap.get("wavetime")
        self.win_wave = snap.get("winWave")
        self.cores = list(snap.get("cores") or [])
        self.player = snap.get("player")

        ship = snap.get("ship") or {}
        self.ship = ship if ship.get("alive") else None

        center = None
        if ship.get("alive"):
            center = (float(ship.get("x") or 0.0), float(ship.get("y") or 0.0))
        elif self.player:
            center = (float(self.player.get("x") or 0.0), float(self.player.get("y") or 0.0))
        vision = float(ship.get("vision") or DEFAULT_VISION)

        current_ores: dict[tuple[int, int], str] = {}
        for ore in snap.get("ores") or []:
            try:
                current_ores[(int(ore["x"]), int(ore["y"]))] = str(ore["i"])
            except (KeyError, TypeError, ValueError):
                continue

        current_blocks: dict[tuple[int, int], tuple[str, int, int]] = {}
        for block in snap.get("blocks") or []:
            try:
                current_blocks[(int(block["x"]), int(block["y"]))] = (
                    str(block["b"]), int(block.get("team", -1)), int(block.get("rot", 0))
                )
            except (KeyError, TypeError, ValueError):
                continue

        if center is None:
            # no ship and no player: the mod still scanned from (0,0), so only
            # accept the data, never invalidate what we knew
            self._ingest(center, vision, current_ores, current_blocks, delta)
            return delta

        self._invalidate(center, vision, current_ores, current_blocks, delta)
        self._ingest(center, vision, current_ores, current_blocks, delta)
        self._update_enemies(snap.get("enemies") or [], delta, center, vision)
        return delta

    def _invalidate(
        self,
        center: tuple[float, float],
        vision: float,
        current_ores: dict[tuple[int, int], str],
        current_blocks: dict[tuple[int, int], tuple[str, int, int]],
        delta: MemoryDelta,
    ) -> None:
        cx, cy = center
        for key in list(self.ores):
            if key in current_ores:
                continue
            if self._tile_visible(key[0], key[1], cx, cy, vision):
                # it was visible a moment ago and is gone now: it was mined out
                del self.ores[key]
                delta.ores_removed.append(key)

        for key in list(self.blocks):
            if key in current_blocks:
                continue
            if self._tile_visible(key[0], key[1], cx, cy, vision):
                del self.blocks[key]
                delta.blocks_removed.append(key)

    def _ingest(
        self,
        center: tuple[float, float] | None,
        vision: float,
        current_ores: dict[tuple[int, int], str],
        current_blocks: dict[tuple[int, int], tuple[str, int, int]],
        delta: MemoryDelta,
    ) -> None:
        now = self.now

        for key, item in current_ores.items():
            old = self.ores.get(key)
            if old is None:
                entry = OreTile(key[0], key[1], item, now)
                self.ores[key] = entry
                delta.ores_added.append(entry)
            else:
                old.item = item
                old.last_seen = now
                old.sightings += 1

        for key, (name, team, rot) in current_blocks.items():
            old = self.blocks.get(key)
            if old is None:
                entry = BlockTile(key[0], key[1], name, team, rot, now)
                self.blocks[key] = entry
                delta.blocks_added.append(entry)
            else:
                if (old.block, old.team, old.rot) != (name, team, rot):
                    old.block, old.team, old.rot = name, team, rot
                    delta.blocks_changed.append(old)
                old.last_seen = now
                old.sightings += 1

        if center is None:
            return

        # mark everything the ship can see as explored
        cx, cy = center
        radius = int(math.ceil(vision / TILE))
        tx = int(cx // TILE)
        ty = int(cy // TILE)
        before = len(self.seen)
        for x in range(tx - radius, tx + radius + 1):
            if self.map_w and not (0 <= x < self.map_w):
                continue
            for y in range(ty - radius, ty + radius + 1):
                if self.map_h and not (0 <= y < self.map_h):
                    continue
                if (x, y) in self.seen:
                    continue
                px, py = _tile_center(x, y)
                if _dist(px, py, cx, cy) <= vision:
                    self.seen.add((x, y))
        delta.new_tiles = len(self.seen) - before

    def _update_enemies(
        self,
        seen: Iterable[dict[str, Any]],
        delta: MemoryDelta,
        center: tuple[float, float],
        vision: float,
    ) -> None:
        now = self.now
        self.enemies = [e for e in self.enemies if now - e.last_seen <= ENEMY_TTL]

        matched: set[int] = set()
        for raw in seen:
            try:
                track = EnemyTrack(
                    type=str(raw.get("t")),
                    team=int(raw.get("team", -1)),
                    x=float(raw.get("x") or 0.0),
                    y=float(raw.get("y") or 0.0),
                    health=float(raw.get("h") or 0.0),
                    last_seen=now,
                    player=bool(raw.get("player")),
                )
            except (TypeError, ValueError):
                continue

            best_index = -1
            best_d = ENEMY_MERGE_PX
            for i, old in enumerate(self.enemies):
                if i in matched:
                    continue
                if old.team != track.team or old.type != track.type:
                    continue
                d = _dist(old.x, old.y, track.x, track.y)
                if d <= best_d:
                    best_index, best_d = i, d

            if best_index < 0:
                self.enemies.append(track)
                delta.new_enemies.append(track)
                matched.add(len(self.enemies) - 1)
            else:
                best = self.enemies[best_index]
                best.x, best.y, best.health = track.x, track.y, track.health
                best.last_seen = now
                best.sightings += 1
                matched.add(best_index)

        # enemies that should be visible right now but were not reported are gone
        keep: list[EnemyTrack] = []
        for i, e in enumerate(self.enemies):
            if i not in matched and self._point_visible(e.x, e.y, center[0], center[1], vision):
                continue
            keep.append(e)
        self.enemies = keep

    @staticmethod
    def _tile_visible(tx: int, ty: int, cx: float, cy: float, vision: float) -> bool:
        px, py = _tile_center(tx, ty)
        return _dist(px, py, cx, cy) <= vision

    @staticmethod
    def _point_visible(x: float, y: float, cx: float, cy: float, vision: float) -> bool:
        return _dist(x, y, cx, cy) <= vision

    # ----------------------------------------------------------------- queries

    def age(self, entry: OreTile | BlockTile | EnemyTrack) -> float:
        return max(0.0, self.now - entry.last_seen)

    def ore_counts(self) -> Counter:
        counts: Counter = Counter()
        for ore in self.ores.values():
            counts[ore.item] += 1
        return counts

    def ore_tiles(self, item: str | None = None) -> list[OreTile]:
        return [o for o in self.ores.values() if item is None or o.item == item]

    def nearest_ore(
        self,
        item: str | None,
        x: float,
        y: float,
        max_distance: float | None = None,
    ) -> OreTile | None:
        best = None
        best_d = math.inf
        for ore in self.ores.values():
            if item is not None and ore.item != item:
                continue
            d = _dist(ore.x * TILE + 4, ore.y * TILE + 4, x, y)
            if d < best_d and (max_distance is None or d <= max_distance):
                best, best_d = ore, d
        return best

    def blocks_of(
        self,
        block: str | None = None,
        team: int | None = None,
    ) -> list[BlockTile]:
        return [
            b for b in self.blocks.values()
            if (block is None or b.block == block) and (team is None or b.team == team)
        ]

    def explored_count(self) -> int:
        return len(self.seen)

    def explored_fraction(self) -> float:
        total = self.map_w * self.map_h
        return (len(self.seen) / total) if total else 0.0

    def frontier(self, limit: int = 64) -> list[tuple[int, int]]:
        """Unseen tiles next to explored ground: where scouting is worthwhile."""
        found: list[tuple[int, int]] = []
        marked: set[tuple[int, int]] = set()
        for x, y in self.seen:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in self.seen or (nx, ny) in marked:
                    continue
                if self.map_w and not (0 <= nx < self.map_w and 0 <= ny < self.map_h):
                    continue
                marked.add((nx, ny))
                found.append((nx, ny))
                if len(found) >= limit:
                    return found
        return found

    def enemies_recent(self, max_age: float = ENEMY_TTL) -> list[EnemyTrack]:
        return [e for e in self.enemies if self.age(e) <= max_age]

    def summary(self) -> str:
        counts = self.ore_counts()
        ore_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        total = self.map_w * self.map_h
        ship = "-"
        if self.ship:
            ship = f"({self.ship.get('x', 0):.0f},{self.ship.get('y', 0):.0f})"
        return (
            f"wave={self.wave} ship={ship} "
            f"ores[{ore_text}] known_ores={len(self.ores)} "
            f"blocks={len(self.blocks)} "
            f"explored={len(self.seen)}/{total} ({self.explored_fraction() * 100:.1f}%) "
            f"enemies={len(self.enemies)}"
        )

    # ------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        def age(entry: OreTile | BlockTile | EnemyTrack) -> float:
            return round(max(0.0, self.now - entry.last_seen), 2)

        return {
            "version": STATE_VERSION,
            "saved_at": round(time.time(), 2),
            "map": {"w": self.map_w, "h": self.map_h},
            "world": {
                "team": self.team,
                "wave": self.wave,
                "wavetime": self.wavetime,
                "winWave": self.win_wave,
                "game_time": self.game_time,
                "snapshots": self.snapshots,
            },
            "ship": self.ship,
            "cores": self.cores,
            "player": self.player,
            "ores": [
                [o.x, o.y, o.item, age(o), o.sightings] for o in self.ores.values()
            ],
            "blocks": [
                [b.x, b.y, b.block, b.team, b.rot, age(b), b.sightings]
                for b in self.blocks.values()
            ],
            "enemies": [
                [e.type, e.team, e.x, e.y, e.health, age(e), e.sightings, e.player]
                for e in self.enemies
            ],
            "seen": self._encode_seen(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldMemory":
        mem = cls()
        now = time.monotonic()
        mem.now = now

        world = data.get("map") or {}
        mem.map_w = int(world.get("w") or 0)
        mem.map_h = int(world.get("h") or 0)

        state = data.get("world") or {}
        mem.team = state.get("team")
        mem.wave = state.get("wave")
        mem.wavetime = state.get("wavetime")
        mem.win_wave = state.get("winWave")
        mem.game_time = float(state.get("game_time") or 0.0)
        mem.snapshots = int(state.get("snapshots") or 0)
        mem.ship = data.get("ship")
        mem.cores = list(data.get("cores") or [])
        mem.player = data.get("player")

        for row in data.get("ores") or []:
            try:
                x, y, item, age, sightings = row
                mem.ores[(int(x), int(y))] = OreTile(
                    int(x), int(y), str(item), now - float(age), int(sightings)
                )
            except (TypeError, ValueError):
                continue

        for row in data.get("blocks") or []:
            try:
                x, y, block, team, rot, age, sightings = row
                mem.blocks[(int(x), int(y))] = BlockTile(
                    int(x), int(y), str(block), int(team), int(rot),
                    now - float(age), int(sightings),
                )
            except (TypeError, ValueError):
                continue

        for row in data.get("enemies") or []:
            try:
                etype, team, x, y, health, age, sightings, player = row
                mem.enemies.append(EnemyTrack(
                    str(etype), int(team), float(x), float(y), float(health),
                    now - float(age), int(sightings), bool(player),
                ))
            except (TypeError, ValueError):
                continue

        mem._decode_seen(data.get("seen") or {})
        return mem

    def _encode_seen(self) -> dict[str, str]:
        rows: dict[int, bytearray] = {}
        width = max(1, self.map_w)
        for x, y in self.seen:
            buf = rows.get(y)
            if buf is None:
                buf = bytearray((width + 7) // 8)
                rows[y] = buf
            buf[x >> 3] |= 1 << (x & 7)
        return {str(y): bytes(buf).hex() for y, buf in rows.items()}

    def _decode_seen(self, encoded: dict[str, str]) -> None:
        for y_text, hex_text in encoded.items():
            try:
                y = int(y_text)
                buf = bytes.fromhex(hex_text)
            except (TypeError, ValueError):
                continue
            for byte_index, byte in enumerate(buf):
                bits = byte
                while bits:
                    low = bits & -bits
                    bit = low.bit_length() - 1
                    x = byte_index * 8 + bit
                    if not self.map_w or x < self.map_w:
                        self.seen.add((x, y))
                    bits ^= low

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "WorldMemory":
        source = Path(path)
        if not source.exists():
            return cls()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", source, exc)
            return cls()
        if int(data.get("version") or 0) != STATE_VERSION:
            log.warning("ignoring memory with version %s", data.get("version"))
            return cls()
        return cls.from_dict(data)


# ---------------------------------------------------------------------- live watch


async def _wait_session(bridge: Bridge, wait: float) -> Any:
    while True:
        try:
            return await bridge.wait_connected(min(wait, 5.0))
        except asyncio.TimeoutError:
            wait -= 5.0
            if wait <= 0:
                raise


async def watch(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    memory = WorldMemory() if args.reset else WorldMemory.load(state_path)
    if args.reset:
        print("starting with empty memory")
    elif memory.ores or memory.blocks:
        print("resumed memory:", memory.summary())
    else:
        print("no saved memory yet")

    bridge = Bridge(host=args.host, port=args.port)
    await bridge.start()
    print(f"listening on {args.host}:{args.port} — feed it snapshots (Ctrl+C to stop)")

    hello_seen = False
    try:
        while True:
            try:
                session = await _wait_session(bridge, args.wait)
            except asyncio.TimeoutError:
                print("no connection from the game")
                return 2

            if not hello_seen:
                print("link up from", session.peer)
                print("hello:", json.dumps(session.hello, sort_keys=True))
                hello_seen = True

            try:
                while True:
                    snap = await bridge.snapshot()
                    delta = memory.update(snap)
                    if not snap.get("ready"):
                        print(f"state={snap.get('state')} — waiting for a loaded world")
                    else:
                        print(f"{memory.summary()}  |  {delta.describe()}")
                        memory.save(state_path)
                        if args.once:
                            return 0
                    await asyncio.sleep(args.interval)
            except BridgeError as exc:
                print("bridge hiccup:", exc)
                memory.save(state_path)
                await asyncio.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        memory.save(state_path)
        print("memory saved to", state_path)
        await bridge.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accumulate AI Ship perception memory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wait", type=float, default=300.0, help="seconds to wait for the mod")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between snapshots")
    parser.add_argument("--state", default=str(DEFAULT_STATE), help="memory file to load/save")
    parser.add_argument("--reset", action="store_true", help="start from an empty memory")
    parser.add_argument("--once", action="store_true", help="take one snapshot, save, and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(watch(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
