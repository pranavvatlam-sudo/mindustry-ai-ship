# AI Ship

An AI companion craft for Mindustry. The ship is a custom unit added by this mod;
its pilot is a Python agent running outside the game.

The design constraint that drives everything: **the mod is dumb**. It exposes a
snapshot, an `eval` hook, pause control, and a waypoint. All strategy and
tile-level decisions live in Python, so the mod rarely needs to be redeployed.

```
Mindustry (Google Play Games Android VM)
  AI Ship mod ── snapshot / eval / pause / waypoints ──┐
                                                       │ TCP, mod dials out
                                                       ▼
                                          Python agent on the host
```

## Current status

Milestone **M0**: link, snapshot, eval. The ship spawns next to your core,
reports only what its sensors can see, and can be steered.

## Repo layout

```
mod.hjson              mod metadata (must stay at the repo root)
scripts/main.js        the bridge + the ship definition
sprites/               ship + weapon sprites
bundles/               localized names
icon.png               mod icon
tools/make_sprites.py  regenerates the sprites
agent/ai_ship/rpc.py   JSON-lines bridge (server side)
agent/ai_ship/probe.py M0 probe CLI
```

## Install in the game

The mod is installed through Mindustry's own GitHub importer, so no adb and no
file copying is needed.

1. Launch Mindustry from Google Play Games.
2. **Mods → Import Mod → GitHub**
3. Type the repo path exactly: `pranavvatlam-sudo/mindustry-ai-ship`
4. Wait for the download, then **restart the game** when prompted.
5. Check that **AI Ship** is in the mod list and enabled.

The importer fetches the latest commit on `main`. To update later, import the
same repo again.

## Run the probe

From the `agent` directory, before or after launching the game:

```powershell
cd agent
python -m ai_ship.probe --wait 300
```

The mod retries the host every few seconds, so ordering does not matter. On
connection you should see a hello, a snapshot summary, and (optionally) an eval.

Useful flags:

```powershell
python -m ai_ship.probe --eval "Vars.state.wave"
python -m ai_ship.probe --chat "AI Ship online"
```

### Networking

The mod tries these hosts in order and reports the one that worked:

1. `10.0.2.2` — the host's loopback alias inside the Play Games Android VM
   (slirp). This is the expected path.
2. `192.168.68.72` — the host's LAN address, as a fallback.
3. `127.0.0.1` — inside the guest; only useful in unusual setups.

Windows may prompt to allow Python through the firewall the first time. Allow
private networks.

## Save safety

`agent` never touches the game's files, but the mod can back up the save folder
from inside the VM:

```powershell
python -m ai_ship.probe --eval "backupSaves()"
```

This copies `saves/` to `saves-ai-backup/` in the game's data directory. Do this
**before** entering a campaign sector if you care about the existing progress
for that sector.

## Developing the mod

Everything at the repo root is shipped to the game as the mod. After editing
`scripts/main.js` or sprites:

```powershell
git add -A
git commit -m "describe change"
git push
```

Then in game: **Mods → Import Mod → GitHub → same repo → restart**.

The Python side can be changed freely without touching the mod, because actions
are executed through `eval` in the mod's script scope.

## Bridge protocol

Requests are one JSON object per line; the mod replies with the same `id`.

| op | fields | returns |
| --- | --- | --- |
| `ping` | – | `{pong, host, time}` |
| `snapshot` | – | game state, vision-limited |
| `eval` | `code` | result of the JS expression |
| `pause` | `paused` | `true` |
| `chat` | `text` | `true` |
| `backup` | – | `{path, files}` |
| `spawn_ship` | – | `true` |
| `ship_move` | `x`, `y` | `true` |
| `ship_hold` | – | `true` |

Helpers available to `eval` inside the mod:

- `placePaid(name, tileX, tileY, rot)` — place a block, charge the core, enforce ship range
- `placeFree(name, tileX, tileY, rot)` — same without cost (debug)
- `breakAt(tileX, tileY)`
- `shipMove(x, y)`, `shipHold()`
- `setPaused(bool)`, `backupSaves()`
- `snapshot()`, `coreBuild()`, `itemMap(core.items)`

## Roadmap

1. **M0** link, snapshot, ship spawned and steerable
2. **M1** perception memory (explored grid), RPC client, save guard
3. **M2** executor: drills → conveyors → core, Duo nests with ammo
4. **M3** planner: LLM goals, validated actions
5. **M4** dashboard, wave-10 Ground Zero run
