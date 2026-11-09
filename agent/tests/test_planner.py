"""Offline unit tests for the M3 planner. No game or network required."""

from __future__ import annotations

import asyncio
import unittest

from ai_ship.memory import WorldMemory
from ai_ship.planner import (
    DEFAULT_MODEL,
    FallbackPlanner,
    HeuristicPlanner,
    LLMPlanner,
    ModelFallbackTransport,
    PlanValidationError,
    PlannerError,
    Policy,
    extract_content,
    parse_plan_json,
    validate_plan,
)


def snapshot(
    *,
    ship=(160.0, 160.0),
    vision=120.0,
    ores=None,
    blocks=None,
    cores=None,
    wave=10,
):
    return {
        "ready": True,
        "state": "playing",
        "paused": False,
        "time": 100.0,
        "team": 1,
        "wave": wave,
        "wavetime": 120.0,
        "winWave": 10,
        "map": {"w": 64, "h": 64},
        "ship": {
            "alive": True,
            "enabled": True,
            "x": ship[0],
            "y": ship[1],
            "health": 950,
            "maxHealth": 950,
            "range": 240,
            "vision": vision,
        },
        "shipCount": 1,
        "player": None,
        "cores": cores
        if cores is not None
        else [{"tx": 22, "ty": 20, "x": 180, "y": 164,
               "health": 1100, "maxHealth": 1100, "items": {"copper": 4000}}],
        "ores": ores
        if ores is not None
        else [{"i": "copper", "x": 10, "y": 10},
              {"i": "lead", "x": 40, "y": 40}],
        "blocks": blocks
        if blocks is not None
        else [{"b": "conveyor", "x": 30, "y": 30, "team": 1, "rot": 1},
              {"b": "conveyor", "x": 50, "y": 50, "team": 2, "rot": 1}],
        "enemies": [],
    }


def memory_from(**kwargs) -> WorldMemory:
    mem = WorldMemory()
    mem.update(snapshot(**kwargs))
    return mem


class ParseTests(unittest.TestCase):
    def test_bare_array(self):
        steps, rationale = parse_plan_json('[{"op":"hold"}]')
        self.assertEqual(steps, [{"op": "hold"}])
        self.assertEqual(rationale, "")

    def test_fenced_object(self):
        text = '```json\n{"rationale":"scout","steps":[{"op":"hold"}]}\n```'
        steps, rationale = parse_plan_json(text)
        self.assertEqual(steps, [{"op": "hold"}])
        self.assertEqual(rationale, "scout")

    def test_prose_around_json(self):
        text = 'Here is the plan:\n{"steps":[{"op":"hold"}]}\nGood luck.'
        steps, _ = parse_plan_json(text)
        self.assertEqual(steps, [{"op": "hold"}])

    def test_reasoning_prose_with_bracket_fragments(self):
        # a reasoning model thought out loud first; the old parser grabbed the
        # first "[" and failed
        text = (
            "The own-building bounds are x [108, 178], y [38, 100].\n"
            "The ship is at (146, 52).\n"
            'Final answer: {"rationale":"defend","steps":[{"op":"hold"}]}'
        )
        steps, rationale = parse_plan_json(text)
        self.assertEqual(steps, [{"op": "hold"}])
        self.assertEqual(rationale, "defend")

    def test_array_inside_prose(self):
        text = 'Plan: [{"op":"move","x":10,"y":20},{"op":"hold"}] done'
        steps, _ = parse_plan_json(text)
        self.assertEqual(len(steps), 2)

    def test_missing_steps(self):
        with self.assertRaises(PlannerError):
            parse_plan_json('{"rationale":"nothing"}')

    def test_extract_content_parts(self):
        payload = {"choices": [{"message": {"content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
        ]}}]}
        self.assertEqual(extract_content(payload), "ab")

    def test_extract_reasoning_fallback(self):
        payload = {"choices": [{"message": {
            "content": None, "reasoning": '{"steps":[{"op":"hold"}]}'
        }}]}
        self.assertEqual(extract_content(payload), '{"steps":[{"op":"hold"}]}')

    def test_extract_empty_message(self):
        with self.assertRaisesRegex(PlannerError, "empty message"):
            extract_content({"choices": [{"message": {"content": ""}}]})

    def test_extract_error(self):
        with self.assertRaises(PlannerError):
            extract_content({"error": {"message": "rate limited"}})


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.mem = memory_from()
        self.policy = Policy()

    def test_accepts_drill_and_duo(self):
        steps = validate_plan([
            {"op": "build", "block": "drill", "tx": 10, "ty": 10, "rot": 0},
            {"op": "build", "block": "duo", "tx": 12, "ty": 12, "rot": 1},
            {"op": "hold"},
        ], self.mem, self.policy)
        self.assertEqual(steps[0]["block"], "mechanical-drill")
        self.assertEqual(steps[1]["rot"], 1)
        self.assertEqual(steps[2], {"op": "hold"})

    def test_rejects_break(self):
        with self.assertRaisesRegex(PlanValidationError, "not allowed to break"):
            validate_plan([{"op": "break", "tx": 30, "ty": 30}], self.mem, self.policy)

    def test_rejects_pause(self):
        with self.assertRaisesRegex(PlanValidationError, "not allowed to pause"):
            validate_plan([{"op": "pause", "paused": True}], self.mem, self.policy)

    def test_rejects_occupied_tile(self):
        # the enemy conveyor at (50,50) is not protected, so this hits the
        # occupancy check rather than the own-building check
        with self.assertRaisesRegex(PlanValidationError, "already has conveyor"):
            validate_plan([
                {"op": "build", "block": "duo", "tx": 50, "ty": 50}
            ], self.mem, self.policy)

    def test_rejects_own_building_tile(self):
        with self.assertRaisesRegex(PlanValidationError, "own building or core"):
            validate_plan([
                {"op": "build", "block": "duo", "tx": 30, "ty": 30}
            ], self.mem, self.policy)

    def test_rejects_near_own_building(self):
        # (28,30) is empty but 2 tiles from the conveyor at (30,30)
        with self.assertRaisesRegex(PlanValidationError, "too close"):
            validate_plan([
                {"op": "build", "block": "duo", "tx": 28, "ty": 30}
            ], self.mem, self.policy)

    def test_rejects_drill_off_ore(self):
        with self.assertRaisesRegex(PlanValidationError, "must be on known ore"):
            validate_plan([
                {"op": "build", "block": "mechanical-drill", "tx": 11, "ty": 11}
            ], self.mem, self.policy)

    def test_rejects_block_on_ore(self):
        with self.assertRaisesRegex(PlanValidationError, "would cover"):
            validate_plan([
                {"op": "build", "block": "conveyor", "tx": 10, "ty": 10}
            ], self.mem, self.policy)

    def test_rejects_outside_map(self):
        with self.assertRaisesRegex(PlanValidationError, "outside the map"):
            validate_plan([
                {"op": "build", "block": "duo", "tx": 100, "ty": 100}
            ], self.mem, self.policy)

    def test_rejects_too_far(self):
        far = memory_from(ship=(1900.0, 1900.0))
        with self.assertRaisesRegex(PlanValidationError, "farther than"):
            validate_plan([
                {"op": "build", "block": "drill", "tx": 10, "ty": 10}
            ], far, self.policy)

    def test_rejects_duplicate_tiles(self):
        with self.assertRaisesRegex(PlanValidationError, "used twice"):
            validate_plan([
                {"op": "build", "block": "duo", "tx": 12, "ty": 12},
                {"op": "build", "block": "router", "tx": 12, "ty": 12},
            ], self.mem, self.policy)

    def test_rejects_empty_and_long_plans(self):
        with self.assertRaisesRegex(PlanValidationError, "empty"):
            validate_plan([], self.mem, self.policy)
        too_long = [{"op": "hold"}] * 21
        with self.assertRaisesRegex(PlanValidationError, "limit"):
            validate_plan(too_long, self.mem, self.policy)

    def test_rejects_unknown_op_and_block(self):
        with self.assertRaisesRegex(PlanValidationError, "unknown op"):
            validate_plan([{"op": "nuke"}], self.mem, self.policy)
        with self.assertRaisesRegex(PlanValidationError, "not allowed"):
            validate_plan([
                {"op": "build", "block": "core-shard", "tx": 12, "ty": 12}
            ], self.mem, self.policy)

    def test_rejects_move_outside_map(self):
        with self.assertRaisesRegex(PlanValidationError, "outside the map"):
            validate_plan([{"op": "move", "x": 9999, "y": 10}], self.mem, self.policy)

    def test_accepts_direction_name_as_rotation(self):
        steps = validate_plan([
            {"op": "build", "block": "conveyor", "tx": 12, "ty": 12, "rot": "east"}
        ], self.mem, self.policy)
        self.assertEqual(steps[0]["rot"], 3)


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.replies:
            raise AssertionError("transport called too many times")
        return self.replies.pop(0)


class LLMPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_validated_plan(self):
        transport = FakeTransport([
            '{"rationale":"drill the copper","steps":['
            '{"op":"build","block":"mechanical-drill","tx":10,"ty":10,"rot":0},'
            '{"op":"hold"}]}'
        ])
        planner = LLMPlanner(transport, model="test/model")
        plan = await planner.plan("mine copper", memory_from())

        self.assertEqual(plan.source, "llm")
        self.assertEqual(plan.model, "test/model")
        self.assertEqual(plan.rationale, "drill the copper")
        self.assertEqual([s["op"] for s in plan.steps], ["build", "hold"])
        self.assertEqual(plan.steps[0]["block"], "mechanical-drill")

    async def test_repairs_one_rejected_reply(self):
        transport = FakeTransport([
            '{"steps":[{"op":"break","tx":30,"ty":30}]}',
            '{"steps":[{"op":"hold"}]}',
        ])
        planner = LLMPlanner(transport, repair_attempts=1)
        plan = await planner.plan("test", memory_from())

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(plan.steps, [{"op": "hold"}])
        # the repair message carries the validator's complaint
        second_user = transport.calls[1][0][-1]["content"]
        self.assertIn("rejected by the safety validator", second_user)
        self.assertIn("not allowed to break", second_user)

    async def test_gives_up_after_repairs(self):
        transport = FakeTransport([
            '{"steps":[{"op":"break","tx":30,"ty":30}]}',
            '{"steps":[{"op":"break","tx":30,"ty":30}]}',
        ])
        planner = LLMPlanner(transport, repair_attempts=1)
        with self.assertRaises(PlanValidationError):
            await planner.plan("test", memory_from())

    async def test_default_model_is_set(self):
        self.assertTrue(DEFAULT_MODEL)

    async def test_reports_the_model_that_answered(self):
        class RateLimited:
            async def complete(self, messages, **kwargs):
                raise PlannerError("HTTP 429 rate-limited")

        class Working:
            async def complete(self, messages, **kwargs):
                return '{"steps":[{"op":"hold"}]}'

        transport = ModelFallbackTransport([
            ("primary/model", RateLimited()),
            ("backup/model", Working()),
        ])
        plan = await LLMPlanner(transport, model="primary/model").plan(
            "test", memory_from()
        )
        self.assertEqual(plan.model, "backup/model")


class FallbackTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_skips_rate_limited_model(self):
        class RateLimited:
            async def complete(self, messages, **kwargs):
                raise PlannerError("OpenRouter HTTP 429: rate-limited upstream")

        class Working:
            async def complete(self, messages, **kwargs):
                return '{"steps":[{"op":"hold"}]}'

        transport = ModelFallbackTransport([
            ("limited/model", RateLimited()),
            ("working/model", Working()),
        ])
        text = await transport.complete([{"role": "user", "content": "plan"}])

        self.assertIn("steps", text)
        self.assertEqual(transport.last_model, "working/model")

    async def test_does_not_retry_auth_errors(self):
        class BadKey:
            async def complete(self, messages, **kwargs):
                raise PlannerError("OpenRouter HTTP 401: invalid key")

        class Working:
            async def complete(self, messages, **kwargs):
                return '{"steps":[{"op":"hold"}]}'

        transport = ModelFallbackTransport([
            ("bad/key", BadKey()),
            ("working/model", Working()),
        ])
        with self.assertRaisesRegex(PlannerError, "401"):
            await transport.complete([])


class FallbackPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_tries_next_model_after_bad_output(self):
        class Garbage:
            async def complete(self, messages, **kwargs):
                return "Let me think about the best tile. The bounds are [108, 178]..."

        class Working:
            async def complete(self, messages, **kwargs):
                return '{"rationale":"ok","steps":[{"op":"hold"}]}'

        planner = FallbackPlanner(
            [("bad/model", Garbage()), ("good/model", Working())],
            repair_attempts=0,
        )
        plan = await planner.plan("test", memory_from())

        self.assertEqual(plan.model, "good/model")
        self.assertEqual(plan.steps, [{"op": "hold"}])

    async def test_stops_on_auth_error(self):
        class BadKey:
            async def complete(self, messages, **kwargs):
                raise PlannerError("OpenRouter HTTP 401: invalid key")

        class Working:
            async def complete(self, messages, **kwargs):
                return '{"steps":[{"op":"hold"}]}'

        planner = FallbackPlanner(
            [("bad/key", BadKey()), ("good/model", Working())],
            repair_attempts=0,
        )
        with self.assertRaisesRegex(PlannerError, "401"):
            await planner.plan("test", memory_from())


class HeuristicTests(unittest.TestCase):
    def test_builds_drill_on_free_copper(self):
        plan = HeuristicPlanner().plan("mine", memory_from())
        builds = [s for s in plan.steps if s["op"] == "build"]
        self.assertTrue(builds)
        self.assertEqual(builds[0]["block"], "mechanical-drill")
        self.assertEqual((builds[0]["tx"], builds[0]["ty"]), (10, 10))
        self.assertEqual(plan.steps[-1], {"op": "hold"})

    def test_scouts_when_no_ore(self):
        mem = memory_from(ores=[], vision=40.0)
        plan = HeuristicPlanner().plan("explore", mem)
        moves = [s for s in plan.steps if s["op"] == "move"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(plan.steps[-1], {"op": "hold"})

    def test_builds_duo_when_no_ore(self):
        mem = memory_from(ores=[])
        plan = HeuristicPlanner().plan("defend this position", mem)
        builds = [s for s in plan.steps if s["op"] == "build"]
        self.assertTrue(builds)
        self.assertEqual(builds[0]["block"], "duo")

    def test_plan_always_validates(self):
        mem = memory_from()
        policy = Policy()
        plan = HeuristicPlanner(policy).plan("mine", mem)
        validate_plan(plan.steps, mem, policy)


if __name__ == "__main__":
    unittest.main()
