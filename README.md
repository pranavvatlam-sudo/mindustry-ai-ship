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

Milestone **M0**: link, snapshot, eval — done.
Milestone **M1**: perception memory — done. The agent accumulates ore tiles, blocks,
enemy tracks, and explored tiles from vision snapshots; anything that leaves
vision and was visible is forgotten (mined or destroyed).
Milestone **M2**: executor — done. It places blocks with real item costs, moves
the ship into range, confirms each placement, and refuses to demolish own
buildings or cores.
Milestone **M3**: planner — done. An LLM (Gemini by default, plus Groq,
OpenRouter or local Ollama) or a deterministic heuristic fallback proposes a
short plan; a strict validator rejects anything unsafe before the executor sees
it. 81 unit tests pass.

Next: **M4** dashboard and a full wave-10 Ground Zero run.

It auto-spawns only in **Ground Zero** (`sector.preset.name == "groundZero"`), so
loading your other sectors does not add modded units to them. Anywhere else, the
agent can create it with the `spawn_ship` op.

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
agent/ai_ship/memory.py M1 perception memory + live watch
agent/ai_ship/executor.py M2 build/drill/conveyor executor
agent/ai_ship/planner.py M3 LLM/heuristic planner + safety validator
agent/tests/           unit tests
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
python -m ai_ship.probe --backup            # copy saves/ before touching a sector
python -m ai_ship.probe --spawn             # spawn the ship in the current game
python -m ai_ship.probe --eval "Vars.state.wave"
python -m ai_ship.probe --chat "AI Ship online"
python -m ai_ship.memory              # accumulate perception memory from snapshots (Ctrl+C to stop)
python -m ai_ship.memory --once       # take one snapshot, save memory, and exit
```

The flags can be combined, e.g. `--backup --spawn`.

## Plan with the agent (M3)

The planner turns a goal into a short list of steps. It is dry-run by default:

```powershell
# no game connection: use the last saved perception memory
python -m ai_ship.planner --offline --heuristic

# ask OpenRouter for a plan, still a dry run
python -m ai_ship.planner --goal "Build a Duo turret near the ship, then hold"

# execute a validated plan (spends real items)
python -m ai_ship.planner --goal "..." --run --yes
```

Set one API key and the planner picks the provider automatically. Any
OpenAI-compatible endpoint works; `--list-providers` shows the built-ins:

```powershell
python -m ai_ship.planner --list-providers

# Gemini free tier (primary recommendation)
$env:GEMINI_API_KEY = "..."      # from https://aistudio.google.com/apikey
$env:AI_PROVIDER = "gemini"      # optional; inferred from the key otherwise
python -m ai_ship.planner --goal "Build a Duo turret near the ship, then hold"

# swap providers without touching code
$env:GROQ_API_KEY = "gsk_..."    # https://console.groq.com/keys
python -m ai_ship.planner --provider groq --goal "..."
```

| Provider | Env key | Default model | Notes |
| --- | --- | --- | --- |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `gemini-3.8-flash` | Free tier, no card. Falls back through `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`. |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Free tier ~1,000 requests/day. |
| `openrouter` | `OPENROUTER_API_KEY` | `google/gemma-4-31b-it:free` | 50 free requests/day; free models rate-limit often. |
| `ollama` | none | `qwen3:8b` | Local `http://localhost:11434/v1`, unlimited. |
| `custom` | `AI_API_KEY` | `AI_MODEL` | Set `AI_BASE_URL` to any `/v1` endpoint (Cloudflare, LM Studio, a proxy). |

Provider selection order: `--provider`, then `$AI_PROVIDER`, then `$AI_BASE_URL`
(custom), then the first key found among `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`; otherwise OpenRouter.

`AI_MODEL` / `--model` sets the primary model, `AI_FALLBACK_MODELS` /
`--fallback-models` the retry list, and `--no-model-fallback` uses only the
primary. When the primary is rate-limited (429), missing (404), or returns prose
instead of JSON, the planner walks the fallback list, then finally the offline
heuristic. The heuristic works without a key or network.

The planner also reads `.env` in the repo root or `agent/` (git-ignored), so the
key never has to go into a command line or chat. Copy `.env.example` to `.env`
and fill in `GEMINI_API_KEY`:

```powershell
Copy-Item .env.example .env
notepad .env
```

A 30-second smoke test of the whole chain, no game needed:

```powershell
python -m ai_ship.planner --offline --show-brief --goal "test"
```

Safety is enforced in code, not just in the prompt:

- the planner cannot emit `break` or `pause` steps;
- it cannot build within 2 tiles of any existing own building or the core
  (`--margin` changes this, but the default is deliberately cautious);
- a mechanical drill must sit on a known free ore tile, and no other block may
  cover ore;
- builds must be within 800 px of the ship, inside the map, and the plan is
  capped at 20 steps;
- the executor re-checks occupancy and range against the live game, so a stale
  plan fails instead of overwriting something.

Because the AI treats the whole existing base as protected, it will usually
scout or build in open ground rather than thread a new conveyor through your
current belts.

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
python -m ai_ship.probe --backup
```

(or `--eval "backupSaves()"`). This copies `saves/` to `saves-ai-backup/` in the
game's data directory. Do this **before** entering a campaign sector if you care
about the existing progress for that sector.

The ship auto-spawns only in Ground Zero. Saves for other sectors are never
given modded units unless you explicitly run `--spawn` there.

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

1. ~~**M0** link, snapshot, ship spawned and steerable~~ — done
2. ~~**M1** perception memory (explored grid), live watch, unit tests~~ — done
3. ~~**M2** executor: drills → conveyors → core, Duo nests with ammo~~ — done, including a live paid build in Ground Zero
4. ~~**M3** planner: LLM goals, validated actions~~ — done; provider-agnostic OpenAI-compatible transport (Gemini, Groq, OpenRouter, Ollama, custom), with a deterministic fallback
5. **M4** dashboard, wave-10 Ground Zero run
