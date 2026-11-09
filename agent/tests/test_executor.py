"""Offline unit tests for the M2 executor. No game connection required."""

from __future__ import annotations

import asyncio
import unittest
from ai_ship.executor import BuildStep, Executor, StepResult
from ai_ship.memory import WorldMemory


class FakeBridge:
    """Stand-in for rpc.Bridge: records eval calls, returns canned results."""

    def __init__(self, eval_results=None, snapshot=None) -> None:
        self.calls = []
        self.eval_results = eval_results or {}
        self._snapshot = snapshot or {"ready": True}
        self._blocks = list((snapshot or {}).get("blocks") or [])
        self._ores = list((snapshot or {}).get("ores") or [])
        self._ship_alive = bool((snapshot or {}).get("ship", {}).get("alive"))

    async def snapshot(self, timeout=15.0):
        snap = dict(self._snapshot)
        snap["blocks"] = [dict(b) for b in self._blocks]
        snap["ship"] = dict(self._snapshot.get("ship", {}))
        snap["ship"]["alive"] = self._ship_alive
        return snap

    async def request(self, op, timeout=10.0, **kwargs):
        self.calls.append((op, kwargs))
        code = kwargs.get("code", "")
        if code in self.eval_results:
            return self.eval_results[code]
        if "placePaid(" in code:
            # simulate placement: parse block/tx/ty/rot crudely
            import re as _re
            m = _re.search(r'placePaid\("([^"]+)",\s*(\d+),\s*(\d+),\s*(\d+)\)', code)
            if m:
                self._blocks.append({"b": m.group(1), "x": int(m.group(2)),
                                     "y": int(m.group(3)), "team": 1, "rot": int(m.group(4))})
            return True
        if "breakAt(" in code:
            import re as _re
            m = _re.search(r"breakAt\((\d+),\s*(\d+)\)", code)
            if m:
                self._blocks = [b for b in self._blocks
                                if not (b["x"] == int(m.group(1)) and b["y"] == int(m.group(2)))]
            return True
        if "shipHold" in code:
            return True
        if "shipMove" in code:
            return True
        if "setPaused" in code:
            return True
        return True


class FakeSession:
    def __init__(self):
        self.hello = {"mod": "ai-ship", "version": "0.1.1", "host": "192.168.68.72"}


def snapshot_with_ship(x=1024.0, y=1024.0, range_=240, cores=None):
    return {
        "ready": True,
        "state": "playing",
        "paused": False,
        "time": 100.0,
        "team": 1,
        "wave": 10,
        "wavetime": 120.0,
        "winWave": 10,
        "map": {"w": 256, "h": 256},
        "ship": {
            "alive": True,
            "enabled": True,
            "x": x, "y": y,
            "health": 950, "maxHealth": 950,
            "vision": 208, "range": range_,
        },
        "shipCount": 1,
        "player": None,
        "cores": cores or [{"tileX": 128, "tileY": 55, "x": 1028, "y": 444,
                            "health": 1100, "maxHealth": 1100, "items": {}}],
        "ores": [],
        "blocks": [],
        "enemies": [],
    }


def make_exec(snapshot=None, eval_results=None):
    bridge = FakeBridge(eval_results=eval_results, snapshot=snapshot)
    bridge._session = FakeSession()
    memory = WorldMemory()
    return Executor(bridge, memory), bridge


class BuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_in_range(self):
        exec_, bridge = make_exec(snapshot_with_ship(1024.0, 1024.0))
        # tile (129, 55) center = (1036, 444); ship at (1024,1024) -> dist > range
        # use a tile right next to the ship instead
        res = await exec_.build("conveyor", 128, 128)
        self.assertTrue(res.ok)
        self.assertIn(("eval", {"code": 'placePaid("conveyor", 128, 128, 0)'}),
                      bridge.calls)

    async def test_build_out_of_range_triggers_move(self):
        exec_, bridge = make_exec(snapshot_with_ship(1024.0, 1024.0))

        async def fail_range(tx, ty, margin=4.0):
            return StepResult(op="move", ok=False, error="out of range")

        exec_.move_into_range = fail_range
        res = await exec_.build("conveyor", 200, 200)
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "out of range")
        # no placement attempted since movement failed
        codes = [kw["code"] for op, kw in bridge.calls if op == "eval"]
        self.assertFalse(any("placePaid" in c for c in codes))

    async def test_build_occupied_tile_rejected(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        # put a known block right next to ship so it's in range and occupied
        snap["blocks"] = [{"b": "conveyor", "x": 128, "y": 128, "team": 1, "rot": 0}]
        exec_, bridge = make_exec(snapshot=snap)
        res = await exec_.build("conveyor", 128, 128)
        self.assertFalse(res.ok)
        self.assertIn("occupied", res.error)

    async def test_ship_not_alive(self):
        snap = snapshot_with_ship()
        snap["ship"] = {"alive": False}
        exec_, _ = make_exec(snapshot=snap)
        res = await exec_.build("conveyor", 128, 128)
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "ship not alive")

    async def test_plan_stops_at_first_error(self):
        exec_, bridge = make_exec(snapshot_with_ship(1024.0, 1024.0))

        async def fake_move_into_range(tx, ty, margin=4.0):
            if (tx, ty) == (999, 999):
                return StepResult(op="move", ok=False, error="out of range")
            return await Executor.move_into_range(exec_, tx, ty, margin)

        exec_.move_into_range = fake_move_into_range
        steps = [
            {"op": "build", "block": "conveyor", "tx": 128, "ty": 128},
            {"op": "build", "block": "drill", "tx": 999, "ty": 999},
            {"op": "build", "block": "duo", "tx": 130, "ty": 130},
        ]
        results = await exec_.plan(steps)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)
        # placement of step 3 never attempted
        codes = [kw["code"] for op, kw in bridge.calls if op == "eval"]
        self.assertEqual(sum("placePaid" in c for c in codes), 1)

    async def test_plan_propagates_error(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        snap["ship"] = {"alive": False}
        exec_, _ = make_exec(snapshot=snap)
        steps = [{"op": "build", "block": "conveyor", "tx": 128, "ty": 128}]
        results = await exec_.plan(steps)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)

    async def test_hold_and_pause(self):
        exec_, bridge = make_exec(snapshot_with_ship(1024.0, 1024.0))
        res = await exec_.hold()
        self.assertTrue(res.ok)
        self.assertIn(("eval", {"code": "shipHold()"}), bridge.calls)

        res = await exec_.pause(True)
        self.assertTrue(res.ok)
        self.assertIn(("eval", {"code": 'setPaused(true)'}), bridge.calls)

    async def test_break_refuses_own_building(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        snap["blocks"] = [{"b": "conveyor", "x": 128, "y": 128, "team": 1, "rot": 0}]
        exec_, bridge = make_exec(snapshot=snap)

        res = await exec_.break_at(128, 128)

        self.assertFalse(res.ok)
        self.assertIn("own building", res.error)
        codes = [kw["code"] for op, kw in bridge.calls if op == "eval"]
        self.assertFalse(any("breakAt" in c for c in codes))

    async def test_break_refuses_core(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        snap["blocks"] = [{"b": "core-shard", "x": 128, "y": 128, "team": 1, "rot": 0}]
        exec_, bridge = make_exec(snapshot=snap)

        res = await exec_.break_at(128, 128)

        self.assertFalse(res.ok)
        self.assertIn("core", res.error)

    async def test_break_allows_enemy_building(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        snap["blocks"] = [{"b": "conveyor", "x": 128, "y": 128, "team": 2, "rot": 0}]
        exec_, bridge = make_exec(snapshot=snap)

        res = await exec_.break_at(128, 128)

        self.assertTrue(res.ok)

    async def test_break_force_allows_own_building(self):
        snap = snapshot_with_ship(1024.0, 1024.0)
        snap["blocks"] = [{"b": "conveyor", "x": 128, "y": 128, "team": 1, "rot": 0}]
        exec_, bridge = make_exec(snapshot=snap)

        res = await exec_.break_at(128, 128, force=True)

        self.assertTrue(res.ok)
        codes = [kw["code"] for op, kw in bridge.calls if op == "eval"]
        self.assertTrue(any("breakAt" in c for c in codes))


if __name__ == "__main__":
    unittest.main()
