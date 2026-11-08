"""M0 probe: prove the link, read a snapshot, run a tiny eval.

    python -m ai_ship.probe
    python -m ai_ship.probe --wait 180
    python -m ai_ship.probe --eval "Vars.state.wave"

Run from the ``agent`` directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .rpc import Bridge, BridgeError


def summarize(snap: dict) -> str:
    if not snap.get("ready"):
        return f"not in a game yet (state={snap.get('state')})"

    lines = []
    lines.append(
        "state={state}  wave={wave}  winWave={winWave}  map={w}x{h}".format(
            state=snap.get("state"),
            wave=snap.get("wave"),
            winWave=snap.get("winWave"),
            w=snap.get("map", {}).get("w"),
            h=snap.get("map", {}).get("h"),
        )
    )

    cores = snap.get("cores") or []
    for i, core in enumerate(cores):
        items = ", ".join(f"{k}={v}" for k, v in sorted(core.get("items", {}).items()))
        lines.append(f"core[{i}] tile=({core.get('x')},{core.get('y')}) hp={core.get('health')} items=[{items}]")

    ship = snap.get("ship") or {}
    if ship.get("alive"):
        lines.append(
            "ship=({:.0f},{:.0f}) hp={}/{} vision={} build_range={}".format(
                ship.get("x", 0),
                ship.get("y", 0),
                ship.get("health", 0),
                ship.get("maxHealth", 0),
                ship.get("vision"),
                ship.get("range"),
            )
        )
    else:
        lines.append(f"ship=dead respawn_in={ship.get('respawnAt')}")

    ores = snap.get("ores") or []
    counts: dict[str, int] = {}
    for ore in ores:
        counts[ore["i"]] = counts.get(ore["i"], 0) + 1
    ore_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    lines.append(f"ores_in_vision: {ore_text}")

    blocks = snap.get("blocks") or []
    enemies = snap.get("enemies") or []
    lines.append(f"visible_blocks={len(blocks)}  visible_enemies={len(enemies)}")

    for enemy in enemies[:5]:
        lines.append(
            "  enemy {} at ({:.0f},{:.0f}) hp={:.0f}".format(
                enemy.get("t"), enemy.get("x", 0), enemy.get("y", 0), enemy.get("h", 0)
            )
        )

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    bridge = Bridge(host=args.host, port=args.port)
    await bridge.start()
    print(f"listening on {args.host}:{args.port} — waiting for the game (Ctrl+C to stop)")

    try:
        session = await bridge.wait_connected(timeout=args.wait)
    except asyncio.TimeoutError:
        print("no connection from the game. Check that the mod is enabled and the game is running.")
        await bridge.stop()
        return 2

    print(f"link up from {session.peer}")
    print("hello:", json.dumps(session.hello, sort_keys=True))

    try:
        pong = await bridge.ping()
        print("ping:", pong)

        if args.backup:
            result = await bridge.request("backup")
            print("\nsave backup:", result)

        if args.spawn:
            spawned = await bridge.request("spawn_ship")
            print("\nship spawned:", spawned)

        snap = await bridge.snapshot()
        print("\n--- snapshot ---")
        print(summarize(snap))

        if args.eval:
            print(f"\n--- eval: {args.eval} ---")
            print(await bridge.eval(args.eval))

        if args.chat:
            await bridge.chat(args.chat)
            print(f"\nchat sent: {args.chat}")
    except BridgeError as exc:
        print("bridge error:", exc, file=sys.stderr)

    await bridge.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the AI Ship bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wait", type=float, default=300.0, help="seconds to wait for the mod")
    parser.add_argument("--eval", default="", help="JS expression to run in the game")
    parser.add_argument("--chat", default="", help="message to post in the game chat")
    parser.add_argument("--spawn", action="store_true", help="spawn the AI ship in the current game")
    parser.add_argument("--backup", action="store_true", help="back up the game's saves folder first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
