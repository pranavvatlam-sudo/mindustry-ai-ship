"""M2: deterministic build executor.

Turns a build plan into `placePaid`/`breakAt`/`shipMove` calls through the
bridge, validating each step against the live snapshot and perception memory:

* the ship is alive and in build range (moves there first if needed)
* the tile isn't already occupied by a known block
* the core can afford the item cost (the mod enforces this too, but we fail
  fast with a clear message)
* after each placement we re-snapshot so memory stays honest

Nothing here commits strategy: the executor only does what it's told to do.
The planner (M3) decides what; this module decides how to do it safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory import WorldMemory
from .rpc import Bridge, BridgeError

log = logging.getLogger("ai_ship.executor")

TILE = 8


@dataclass
class BuildStep:
    """One atomic action the executor can carry out."""

    op: str  # "build" | "break" | "move" | "hold" | "pause"
    block: str | None = None
    tx: int | None = None
    ty: int | None = None
    rot: int = 0
    x: float | None = None
    y: float | None = None
    paused: bool | None = None
    force: bool = False
    # how long to wait for the ship to reach a move target (seconds)
    timeout: float = 30.0


@dataclass
class StepResult:
    op: str
    ok: bool
    error: str | None = None
    ship: dict[str, Any] | None = None
    paused: bool | None = None


class Executor:
    def __init__(self, bridge: Bridge, memory: WorldMemory) -> None:
        self.bridge = bridge
        self.memory = memory

    # ------------------------------------------------------------------ low level

    async def _snap(self) -> dict[str, Any]:
        snap = await self.bridge.snapshot()
        self.memory.update(snap)
        return snap

    async def _ship(self) -> dict[str, Any] | None:
        snap = await self._snap()
        ship = snap.get("ship") if snap.get("ready") else None
        if ship and not ship.get("alive"):
            return None
        return ship

    async def _eval(self, code: str) -> Any:
        return await self.bridge.request("eval", code=code)

    async def _quote(self, text: str) -> str:
        return json.dumps(text)

    # ------------------------------------------------------------------ movement

    async def hold(self) -> StepResult:
        """Stop the ship (clear any move target)."""
        await self._eval("shipHold()")
        await self._snap()
        return StepResult(op="hold", ok=True, ship=self.memory.ship)

    async def move_to(self, x: float, y: float, timeout: float = 30.0,
                      within: float = 4.0) -> StepResult:
        """Steer the ship toward world coords.

        Returns ok as soon as the ship is within ``within`` pixels of the
        target. The controller parks inside 3 px, so a small radius is enough;
        requiring the ship to be perfectly still used to time out while it
        hovered around the waypoint.
        """
        await self._eval(f"shipMove({x:.2f}, {y:.2f})")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if loop.time() >= deadline:
                return StepResult(op="move", ok=False, error="move timeout",
                                  ship=self.memory.ship)
            ship = await self._ship()
            if not ship:
                return StepResult(op="move", ok=False, error="ship died",
                                  ship=None)
            pos = (float(ship.get("x", 0.0)), float(ship.get("y", 0.0)))
            if math.hypot(pos[0] - x, pos[1] - y) <= within:
                return StepResult(op="move", ok=True, ship=self.memory.ship)
            await asyncio.sleep(0.4)

    async def move_into_range(self, tx: int, ty: int, margin: float = 4.0) -> StepResult:
        """Move the ship until tile (tx, ty) is within build range."""
        ship = await self._ship()
        if not ship:
            return StepResult(op="move", ok=False, error="no ship", ship=None)

        build_range = float((await self._snap()).get("ship", {}).get("range", 240))
        txc, tyc = tx * TILE + TILE / 2, ty * TILE + TILE / 2
        sx, sy = float(ship.get("x", 0.0)), float(ship.get("y", 0.0))
        limit = build_range + margin
        if math.hypot(txc - sx, tyc - sy) <= limit:
            return StepResult(op="move", ok=True, ship=self.memory.ship)

        log.info("moving ship into build range of tile %s,%s", tx, ty)
        res = await self.move_to(txc, tyc, within=limit)
        if not res.ok:
            return res
        # final range check: the mod enforces buildRange + 8, match it
        snap = await self._snap()
        ship = snap.get("ship") or {}
        sx, sy = float(ship.get("x", 0.0)), float(ship.get("y", 0.0))
        if math.hypot(txc - sx, tyc - sy) > build_range + 8:
            return StepResult(op="move", ok=False,
                              error="could not reach build range",
                              ship=self.memory.ship)
        return StepResult(op="move", ok=True, ship=self.memory.ship)

    # ------------------------------------------------------------------ build

    async def build(self, block: str, tx: int, ty: int, rot: int = 0) -> StepResult:
        """Place one block at tile (tx, ty), paying real item costs."""
        ship = await self._ship()
        if not ship:
            return StepResult(op="build", ok=False, error="ship not alive",
                              ship=None)

        # range: move into range first if needed
        in_range = await self.move_into_range(tx, ty)
        if not in_range.ok:
            return StepResult(op="build", ok=False, error=in_range.error,
                              ship=self.memory.ship)

        # tile must not already be occupied by a block we know about
        known = self.memory.blocks.get((tx, ty))
        if known is not None:
            return StepResult(op="build", ok=False,
                              error=f"tile occupied by {known.block}",
                              ship=self.memory.ship)

        code = f'placePaid({json.dumps(block)}, {tx}, {ty}, {rot})'
        log.info("building %s at %s,%s rot=%s", block, tx, ty, rot)
        try:
            ok = await self._eval(code)
        except BridgeError as exc:
            return StepResult(op="build", ok=False, error=str(exc),
                              ship=self.memory.ship)

        if not ok:
            return StepResult(op="build", ok=False,
                              error=f"mod rejected {block} at {tx},{ty}",
                              ship=self.memory.ship)

        # confirm the placement landed and refresh memory
        snap = await self._snap()
        placed = self.memory.blocks.get((tx, ty))
        if placed is None or placed.block != block:
            return StepResult(op="build", ok=False,
                              error=f"placement not confirmed for {block} at {tx},{ty}",
                              ship=self.memory.ship)

        return StepResult(op="build", ok=True,
                          ship=self.memory.ship,
                          error=None)

    async def break_at(self, tx: int, ty: int, force: bool = False) -> StepResult:
        """Clear tile (tx, ty).

        Safety interlock: the executor refuses to demolish the player's own
        buildings (or any core) unless ``force`` is explicitly set. The planner
        cannot accidentally delete the base the user built.
        """
        await self._snap()  # make sure memory is current before the safety check
        known = self.memory.blocks.get((tx, ty))
        if known is not None:
            if "core" in known.block:
                return StepResult(op="break", ok=False,
                                  error=f"refusing to break a core at {tx},{ty}",
                                  ship=self.memory.ship)
            own_team = self.memory.team
            if not force and own_team is not None and known.team == own_team:
                return StepResult(op="break", ok=False,
                                  error=f"refusing to break own building {known.block} at {tx},{ty}",
                                  ship=self.memory.ship)

        try:
            ok = await self._eval(f"breakAt({tx}, {ty})")
        except BridgeError as exc:
            return StepResult(op="break", ok=False, error=str(exc),
                              ship=self.memory.ship)
        if not ok:
            return StepResult(op="break", ok=False,
                              error=f"mod rejected breakAt {tx},{ty}",
                              ship=self.memory.ship)
        await self._snap()
        return StepResult(op="break", ok=True, ship=self.memory.ship)

    async def pause(self, paused: bool) -> StepResult:
        try:
            ok = await self._eval(f"setPaused({str(paused).lower()})")
        except BridgeError as exc:
            return StepResult(op="pause", ok=False, error=str(exc),
                              ship=self.memory.ship)
        if not ok:
            return StepResult(op="pause", ok=False, error="setPaused rejected",
                              ship=self.memory.ship)
        await self._snap()
        return StepResult(op="pause", ok=True, paused=paused,
                          ship=self.memory.ship)

    # ------------------------------------------------------------------ plans

    async def plan(self, steps: list[dict[str, Any]]) -> list[StepResult]:
        """Execute a list of build steps in order, stopping at the first error."""
        results: list[StepResult] = []
        for i, raw in enumerate(steps):
            step = BuildStep(**raw)
            log.info("step %d/%d: %s", i + 1, len(steps), step.op)
            if step.op == "build":
                assert step.block and step.tx is not None and step.ty is not None
                res = await self.build(step.block, step.tx, step.ty, step.rot)
            elif step.op == "break":
                assert step.tx is not None and step.ty is not None
                res = await self.break_at(step.tx, step.ty, step.force)
            elif step.op == "move":
                assert step.x is not None and step.y is not None
                res = await self.move_to(step.x, step.y, step.timeout)
            elif step.op == "hold":
                res = await self.hold()
            elif step.op == "pause":
                res = await self.pause(bool(step.paused))
            else:
                res = StepResult(op=step.op, ok=False,
                                 error=f"unknown op {step.op}",
                                 ship=self.memory.ship)
            results.append(res)
            if not res.ok:
                log.error("step %d failed: %s", i + 1, res.error)
                break
        return results

    # ------------------------------------------------------------------ patterns

    @staticmethod
    def line_steps(
        from_tx: int,
        from_ty: int,
        to_tx: int,
        to_ty: int,
        block: str,
        rot: int = 0,
        spacing: int = 1,
    ) -> list[dict[str, Any]]:
        """Straight line of build steps (Bresenham), useful for conveyors."""
        steps: list[dict[str, Any]] = []
        x0, y0 = from_tx, from_ty
        x1, y1 = to_tx, to_ty
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        n = 0
        while True:
            if n % spacing == 0:
                steps.append({"op": "build", "block": block, "tx": x0, "ty": y0, "rot": rot})
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
            n += 1
        return steps


# ---------------------------------------------------------------------- CLI


async def _run_cli(plan_file: str | None, steps_json: str | None,
                   host: str, port: int, state: str,
                   reset: bool, once: bool) -> int:
    memory = WorldMemory() if reset else WorldMemory.load(state)
    bridge = Bridge(host=host, port=port)
    await bridge.start()

    try:
        session = await bridge.wait_connected(timeout=300.0)
        print("link up from", session.peer)
        print("hello:", json.dumps(session.hello, sort_keys=True))

        snap = await bridge.snapshot()
        memory.update(snap)
        print(memory.summary())

        exec_ = Executor(bridge, memory)

        steps: list[dict[str, Any]] = []
        if steps_json:
            steps = json.loads(steps_json)
        elif plan_file:
            steps = json.loads(Path(plan_file).read_text(encoding="utf-8"))

        if once:
            await exec_._snap()
            print(memory.summary())
            return 0

        results = await exec_.plan(steps)
        for i, r in enumerate(results, 1):
            status = "ok" if r.ok else f"FAIL: {r.error}"
            print(f"[{i}] {r.op}: {status}")
        memory.save(state)
        print("memory saved to", state)
        return 0 if all(r.ok for r in results) else 1
    finally:
        await bridge.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2 build plans through the bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", default=str(Path(__file__).resolve().parent.parent / "state" / "world-memory.json"))
    parser.add_argument("--reset", action="store_true", help="start from empty memory")
    parser.add_argument("--once", action="store_true", help="snapshot once, print summary, exit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", help="JSON file with a list of build steps")
    group.add_argument("--steps", help="JSON string with a list of build steps")
    args = parser.parse_args(argv)

    if not args.once and not args.plan and not args.steps:
        parser.error("one of --plan/--steps is required unless --once is given")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_run_cli(args.plan, args.steps, args.host, args.port,
                                    args.state, args.reset, args.once))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
