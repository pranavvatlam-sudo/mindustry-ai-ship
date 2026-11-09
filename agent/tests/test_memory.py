"""Unit tests for the M1 perception memory. Standard library only."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from ai_ship.memory import EnemyTrack, WorldMemory


def snapshot(
    *,
    ship: tuple[float, float] = (1024.0, 1024.0),
    ores: tuple[tuple[int, int, str], ...] = (),
    blocks: tuple[tuple[int, int, str, int, int], ...] = (),
    enemies: tuple[tuple[str, int, float, float, float], ...] = (),
    game_time: float = 1.0,
    size: tuple[int, int] = (256, 256),
    wave: int = 10,
    ready: bool = True,
) -> dict:
    return {
        "ready": ready,
        "state": "playing" if ready else "menu",
        "paused": False,
        "time": game_time,
        "map": {"w": size[0], "h": size[1]},
        "team": 1,
        "wave": wave,
        "wavetime": 120.0,
        "winWave": 10,
        "ship": {
            "alive": True,
            "enabled": True,
            "x": ship[0],
            "y": ship[1],
            "health": 950,
            "maxHealth": 950,
            "vision": 208,
            "range": 240,
        },
        "shipCount": 1,
        "player": None,
        "cores": [],
        "own": [],
        "ores": [{"i": item, "x": x, "y": y} for x, y, item in ores],
        "blocks": [
            {"b": name, "x": x, "y": y, "team": team, "rot": rot}
            for x, y, name, team, rot in blocks
        ],
        "enemies": [
            {"t": etype, "team": team, "x": x, "y": y, "h": hp, "player": False}
            for etype, team, x, y, hp in enemies
        ],
    }


class UpdateTests(unittest.TestCase):
    def test_adds_visible_ores_and_blocks(self):
        mem = WorldMemory()
        delta = mem.update(
            snapshot(
                ores=((130, 130, "copper"), (131, 130, "lead")),
                blocks=((131, 131, "conveyor", 1, 0),),
            )
        )

        self.assertEqual(len(mem.ores), 2)
        self.assertEqual(mem.ore_counts()["copper"], 1)
        self.assertEqual(mem.ore_counts()["lead"], 1)
        self.assertEqual(len(mem.blocks), 1)
        self.assertEqual(len(delta.ores_added), 2)
        self.assertEqual(len(delta.blocks_added), 1)
        self.assertGreater(mem.explored_count(), 0)
        self.assertGreater(delta.new_tiles, 0)

    def test_visible_ore_removed_when_mined(self):
        mem = WorldMemory()
        mem.update(snapshot(ores=((130, 130, "copper"),)))

        delta = mem.update(snapshot())

        self.assertEqual(len(mem.ores), 0)
        self.assertEqual(delta.ores_removed, [(130, 130)])

    def test_ore_outside_vision_is_remembered(self):
        mem = WorldMemory()
        mem.update(snapshot(ship=(200.0, 200.0), ores=((10, 10, "lead"),)))
        self.assertEqual(len(mem.ores), 1)

        # move far away; the tile is no longer visible, so it must survive
        delta = mem.update(snapshot(ship=(1800.0, 1800.0)))

        self.assertEqual(len(mem.ores), 1)
        self.assertEqual(delta.ores_removed, [])

    def test_block_change_is_reported(self):
        mem = WorldMemory()
        mem.update(snapshot(blocks=((131, 131, "conveyor", 1, 0),)))

        delta = mem.update(snapshot(blocks=((131, 131, "duo", 1, 0),)))

        self.assertEqual(len(delta.blocks_changed), 1)
        self.assertEqual(mem.blocks[(131, 131)].block, "duo")
        self.assertEqual(delta.blocks_removed, [])

    def test_block_destroyed_when_visible_and_gone(self):
        mem = WorldMemory()
        mem.update(snapshot(blocks=((131, 131, "duo", 1, 0),)))

        delta = mem.update(snapshot())

        self.assertEqual(delta.blocks_removed, [(131, 131)])
        self.assertEqual(len(mem.blocks), 0)

    def test_map_change_resets_memory(self):
        mem = WorldMemory()
        mem.update(snapshot(ores=((130, 130, "copper"),)))

        delta = mem.update(snapshot(size=(128, 128)))

        self.assertTrue(delta.reset)
        self.assertEqual(len(mem.ores), 0)
        self.assertEqual(mem.map_w, 128)

    def test_not_ready_snapshot_is_ignored(self):
        mem = WorldMemory()
        mem.update(snapshot(ores=((130, 130, "copper"),)))

        delta = mem.update(snapshot(ready=False))

        self.assertFalse(delta)
        self.assertEqual(len(mem.ores), 1)


class ExplorationTests(unittest.TestCase):
    def test_explored_area_only_grows_with_movement(self):
        mem = WorldMemory()
        mem.update(snapshot())
        covered = mem.explored_count()

        delta = mem.update(snapshot())
        self.assertEqual(delta.new_tiles, 0)
        self.assertEqual(mem.explored_count(), covered)

        delta = mem.update(snapshot(ship=(1024.0 + 16 * 8, 1024.0)))
        self.assertGreater(delta.new_tiles, 0)
        self.assertGreater(mem.explored_count(), covered)
        self.assertLessEqual(mem.explored_fraction(), 1.0)

    def test_frontier_is_unseen_and_adjacent(self):
        mem = WorldMemory()
        mem.update(snapshot())

        frontier = mem.frontier(limit=20)

        self.assertTrue(frontier)
        for tile in frontier:
            self.assertNotIn(tile, mem.seen)


class QueryTests(unittest.TestCase):
    def test_nearest_ore(self):
        mem = WorldMemory()
        mem.update(
            snapshot(
                ores=(
                    (130, 130, "copper"),
                    (134, 130, "copper"),
                    (140, 140, "lead"),
                )
            )
        )

        nearest = mem.nearest_ore("copper", 1024.0, 1024.0)
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest.pos, (130, 130))

        self.assertIsNone(mem.nearest_ore("titanium", 1024.0, 1024.0))
        self.assertIsNone(mem.nearest_ore("copper", 1024.0, 1024.0, max_distance=1.0))

    def test_blocks_filtered_by_name_and_team(self):
        mem = WorldMemory()
        mem.update(
            snapshot(
                blocks=(
                    (131, 131, "duo", 1, 0),
                    (132, 131, "conveyor", 1, 0),
                    (133, 131, "duo", 2, 0),
                )
            )
        )

        self.assertEqual(len(mem.blocks_of(block="duo")), 2)
        self.assertEqual(len(mem.blocks_of(block="duo", team=1)), 1)
        self.assertEqual(len(mem.blocks_of(team=2)), 1)


class EnemyTests(unittest.TestCase):
    def test_enemy_moves_are_merged_into_one_track(self):
        mem = WorldMemory()
        mem.update(snapshot(enemies=(("dagger", 2, 1064.0, 1024.0, 120.0),)))
        mem.update(snapshot(enemies=(("dagger", 2, 1069.0, 1027.0, 118.0),)))

        self.assertEqual(len(mem.enemies), 1)
        track = mem.enemies[0]
        self.assertEqual(track.sightings, 2)
        self.assertEqual((track.x, track.y), (1069.0, 1027.0))
        self.assertAlmostEqual(track.health, 118.0)

    def test_enemy_gone_from_vision_is_dropped(self):
        mem = WorldMemory()
        mem.update(snapshot(enemies=(("dagger", 2, 1064.0, 1024.0, 120.0),)))

        mem.update(snapshot())

        self.assertEqual(mem.enemies, [])

    def test_enemy_track_ttl(self):
        mem = WorldMemory()
        mem.update(snapshot())
        mem.enemies.append(
            EnemyTrack("dagger", 2, 1900.0, 1900.0, 100.0, time.monotonic() - 1000.0)
        )

        mem.update(snapshot())

        self.assertEqual(mem.enemies, [])


class PersistenceTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        mem = WorldMemory()
        mem.update(
            snapshot(
                ores=((130, 130, "copper"), (140, 140, "lead")),
                blocks=((131, 131, "duo", 1, 2),),
                enemies=(("dagger", 2, 1064.0, 1024.0, 120.0),),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "memory.json"
            mem.save(path)
            self.assertTrue(path.exists())

            loaded = WorldMemory.load(path)

        self.assertEqual(loaded.map_w, mem.map_w)
        self.assertEqual(loaded.ore_counts(), mem.ore_counts())
        self.assertEqual(len(loaded.blocks), 1)
        self.assertEqual(loaded.blocks[(131, 131)].rot, 2)
        self.assertEqual(len(loaded.enemies), 1)
        self.assertEqual(loaded.explored_count(), mem.explored_count())
        self.assertIn("copper=1", loaded.summary())

    def test_load_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = WorldMemory.load(Path(tmp) / "nope.json")

        self.assertEqual(len(loaded.ores), 0)
        self.assertEqual(loaded.explored_count(), 0)


if __name__ == "__main__":
    unittest.main()
