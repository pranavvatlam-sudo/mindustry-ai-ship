// AI Ship — a thin RPC bridge between Mindustry and an external agent.
// The agent lives outside the game. This mod only observes, steers, and executes.

var MOD_VERSION = "0.1.1";
var PORT = 8765;
// 10.0.2.2 is the host loopback alias inside the Google Play Games Android VM.
// The LAN address is the fallback if slirp forwarding is restricted.
var HOSTS = ["10.0.2.2", "192.168.68.72", "127.0.0.1"];
var RECONNECT_MS = 2500;
var VISION_RANGE = 26 * 8;
var RESPAWN_SECONDS = 20;

// ---------------------------------------------------------------- content

var steer = {target: null};
var shipUnit = null;
var shipRespawnAt = 0;
var shipEnabled = false;
var moveVec = new Vec2();
var lastErrorText = null;

// Rhino exceptions inside an update handler kill the whole game, so every
// recurring hook runs through this and only logs each distinct error once.
function noteError(where, err){
    var text = "" + (err && err.message ? err.message : err);
    if(text === lastErrorText) return;
    lastErrorText = text;
    print("[ai-ship] " + where + " error: " + text);
}

var shipController = extend(AIController, {
    updateMovement: function(){
        try{
            var u = shipUnit;
            if(u == null || !u.isAdded()) return;

            var t = steer.target;
            if(t == null){
                // idle: kill any residual velocity so the ship stays put
                u.vel.setZero();
                return;
            }

            var dx = t.x - u.x, dy = t.y - u.y;
            var dist = Math.sqrt(dx * dx + dy * dy);
            if(dist < 3) return;

            // ease off near the waypoint so it does not orbit it forever
            var fac = Math.min(1, dist / 28);
            var sp = u.speed() * fac;

            moveVec.set(dx / dist * sp, dy / dist * sp);
            u.movePref(moveVec);
            u.lookAt(t);
        }catch(err){
            noteError("movement", err);
            steer.target = null;
        }
    }
});

var shipBullet = new BasicBulletType(5.5, 16);
shipBullet.lifetime = 45;
shipBullet.width = 7;
shipBullet.height = 11;
shipBullet.frontColor = Color.valueOf("9ff4ff");
shipBullet.backColor = Color.valueOf("2f6fd0");
shipBullet.shootEffect = Fx.shootSmall;
shipBullet.smokeEffect = Fx.shootSmallSmoke;
shipBullet.ammoMultiplier = 2;

var shipWeapon = new Weapon("ai-ship-weapon");
shipWeapon.bullet = shipBullet;
shipWeapon.reload = 9;
shipWeapon.x = 0;
shipWeapon.y = -2;
shipWeapon.mirror = false;
shipWeapon.shootCone = 20;
shipWeapon.top = true;

var shipType = new UnitType("ai-ship");
shipType.flying = true;
shipType.health = 950;
shipType.armor = 3;
shipType.speed = 3.0;
shipType.accel = 0.11;
shipType.drag = 0.035;
shipType.hitSize = 15;
shipType.rotateSpeed = 6.5;
shipType.omniMovement = true;
shipType.faceTarget = true;
shipType.buildSpeed = 2.5;
shipType.buildRange = 240;
shipType.mineSpeed = 1.5;
shipType.mineTier = 2;
shipType.itemCapacity = 60;
shipType.engineOffset = 9;
shipType.engineSize = 3.2;
shipType.playerControllable = false;
// stay passive until the external agent arms it
shipType.targetAir = false;
shipType.targetGround = false;
// mod sprites are prefixed with the mod name, so the weapon region is
// "<unit type name>-weapon" == "ai-ship-ai-ship-weapon"
shipWeapon.name = shipType.name + "-weapon";
shipType.weapons.add(shipWeapon);
shipType.controller = func(function(u){ return shipController; });

// ---------------------------------------------------------------- bridge state

var started = false;
var running = false;
var connected = false;
var activeHost = null;
var announcedFailure = false;

// ---------------------------------------------------------------- helpers

function say(text){
    print("[ai-ship] " + text);
    // called from the socket thread, so UI work has to hop to the render thread
    try{
        Core.app.post(run(function(){
            try{ Vars.ui.hudfrag.showToast(text); }catch(e){}
            try{ Vars.ui.chatfrag.addMessage("[accent][AI] []" + text); }catch(e){}
        }));
    }catch(e){}
}

function gameTeam(){
    return Vars.player != null ? Vars.player.team() : Vars.state.rules.defaultTeam;
}

function coreBuild(){
    try{
        var cores = gameTeam().data().cores;
        return cores.size > 0 ? cores.first() : null;
    }catch(e){
        return null;
    }
}

function findBlock(name){
    var all = Vars.content.blocks();
    for(var i = 0; i < all.size; i++){
        if(all.get(i).name === name) return all.get(i);
    }
    return null;
}

function findItem(name){
    var all = Vars.content.items();
    for(var i = 0; i < all.size; i++){
        if(all.get(i).name === name) return all.get(i);
    }
    return null;
}

function itemMap(module){
    var out = {};
    if(module == null) return out;
    var all = Vars.content.items();
    for(var i = 0; i < all.size; i++){
        var item = all.get(i);
        var amount = module.get(item);
        if(amount > 0) out[item.name] = amount;
    }
    return out;
}

function tileOf(build){
    try{ return {x: build.tileX(), y: build.tileY()}; }
    catch(e){ return {x: Math.floor(build.x / 8), y: Math.floor(build.y / 8)}; }
}

// Runs fn on the game/render thread and blocks the caller until it finishes.
// Core.app.post still runs while the game is paused, which is what makes
// "inspect everything while frozen" possible.
function onMain(fn, timeoutMs){
    var latch = new Packages.java.util.concurrent.CountDownLatch(1);
    var box = {done: false, value: null, error: null};

    Core.app.post(run(function(){
        try{
            box.value = fn();
        }catch(err){
            box.error = "" + (err && err.message ? err.message : err);
        }finally{
            box.done = true;
            latch.countDown();
        }
    }));

    latch.await(timeoutMs || 8000, Packages.java.util.concurrent.TimeUnit.MILLISECONDS);
    if(!box.done) throw new Error("game thread timeout");
    if(box.error != null) throw new Error(box.error);
    return box.value;
}

function normalize(value){
    if(value === undefined || value === null) return null;
    var type = typeof value;
    if(type === "number" || type === "string" || type === "boolean") return value;
    try{
        if(Array.isArray(value)) return value;
    }catch(e){}
    try{
        return "" + value;
    }catch(e){
        return "?";
    }
}

// ---------------------------------------------------------------- ship

// The ship only auto-spawns in Ground Zero, so loading other sectors or saves
// never adds modded content to them. The agent can still spawn it anywhere
// through the spawn_ship op.
function sectorIsGroundZero(){
    try{
        var sector = Vars.state.getSector();
        return sector != null && sector.preset != null && sector.preset.name === "groundZero";
    }catch(e){
        return false;
    }
}

function shipsOf(team){
    var out = [];
    try{
        var units = team.data().units;
        for(var i = 0; i < units.size; i++){
            var u = units.get(i);
            // mod content names are prefixed (ai-ship-ai-ship), and comparing
            // the numeric content id is the same on every platform
            if(u != null && u.isAdded() && u.type.id === shipType.id) out.push(u);
        }
    }catch(e){ }
    return out;
}

// Keeps exactly one ship alive: the one with id `keep`, or none when keep is null.
function pruneShips(keep){
    var all = shipsOf(gameTeam());
    for(var i = 0; i < all.length; i++){
        var u = all[i];
        if(keep != null && u.id === keep.id) continue;
        try{ u.remove(); }catch(e){ }
    }
}

// Reuses a ship that already exists in the world (for example one loaded from
// a save) instead of spawning a second one.
function adoptShip(){
    var all = shipsOf(gameTeam());
    if(all.length === 0) return false;

    shipUnit = all[0];
    pruneShips(shipUnit);
    try{ shipUnit.controller(shipController); }catch(e){ }
    try{ shipController.unit(shipUnit); }catch(e){ }
    steer.target = null;
    return true;
}

function spawnShip(){
    try{
        if(!Vars.state.isGame()) return null;

        var team = gameTeam();
        var x, y;
        var cores = team.data().cores;

        if(cores.size > 0){
            var core = cores.first();
            x = core.x + 28;
            y = core.y + 8;
        }else if(Vars.player != null){
            x = Vars.player.x + 28;
            y = Vars.player.y;
        }else{
            x = Vars.world.width() * 4;
            y = Vars.world.height() * 4;
        }

        // replace any existing ship(s) so there is never more than one
        pruneShips(null);

        shipUnit = shipType.spawn(team, x, y);
        try{ shipUnit.controller(shipController); }catch(e){ }
        try{ shipController.unit(shipUnit); }catch(e){ }
        pruneShips(shipUnit);
        steer.target = null;
        shipEnabled = true;
        return shipUnit;
    }catch(err){
        print("[ai-ship] spawn failed: " + err);
        return null;
    }
}

function ensureShip(){
    try{
        if(!Vars.state.isGame()) return;
        if(!shipEnabled && sectorIsGroundZero()) shipEnabled = true;
        if(shipUnit != null && shipUnit.isAdded()) return;

        // pick up a ship that survived a save/load, and drop any duplicates,
        // even in a sector that does not auto-spawn it
        if(adoptShip()) return;
        if(!shipEnabled) return;

        if(shipRespawnAt <= 0){
            shipRespawnAt = Time.time + 2;
            return;
        }
        if(Time.time < shipRespawnAt) return;

        shipRespawnAt = 0;
        if(spawnShip() == null){
            shipRespawnAt = Time.time + 10;
        }
    }catch(err){
        noteError("ensureShip", err);
    }
}

// ---------------------------------------------------------------- snapshot

function snapshot(){
    var st = Vars.state;
    var out = {
        mod: MOD_VERSION,
        host: activeHost,
        time: Time.time,
        state: st.isPaused() ? "paused" : (st.isPlaying() ? "playing" : "menu"),
        paused: st.isPaused(),
        ready: false
    };

    if(!st.isGame()) return out;
    if(Vars.world == null || Vars.world.width() <= 0) return out;

    out.ready = true;
    out.wave = st.wave;
    out.wavetime = st.wavetime;
    out.winWave = st.rules.winWave;
    out.map = {w: Vars.world.width(), h: Vars.world.height()};

    var team = gameTeam();
    out.team = team.id;

    var data = team.data();
    var cores = [];
    for(var i = 0; i < data.cores.size; i++){
        var c = data.cores.get(i);
        cores.push({
            x: c.x,
            y: c.y,
            tx: c.tileX(),
            ty: c.tileY(),
            health: c.health,
            maxHealth: c.maxHealth,
            items: itemMap(c.items)
        });
    }
    out.cores = cores;

    // ship
    var ship = {alive: false, enabled: shipEnabled, respawnAt: shipRespawnAt > 0 ? shipRespawnAt : null};
    if(shipUnit != null && shipUnit.isAdded()){
        ship = {
            alive: true,
            enabled: shipEnabled,
            x: shipUnit.x,
            y: shipUnit.y,
            health: shipUnit.health,
            maxHealth: shipUnit.maxHealth,
            range: shipType.buildRange,
            vision: VISION_RANGE,
            target: steer.target != null ? {x: steer.target.x, y: steer.target.y} : null
        };
    }
    out.ship = ship;
    out.shipCount = shipsOf(team).length;
    out.player = Vars.player != null ? {x: Vars.player.x, y: Vars.player.y} : null;

    // only what the ship can actually see
    var vx = ship.alive ? ship.x : (Vars.player != null ? Vars.player.x : 0);
    var vy = ship.alive ? ship.y : (Vars.player != null ? Vars.player.y : 0);

    // own team's buildings the ship can see (plus everything within build range)
    var own = [];
    var ownRange = VISION_RANGE * VISION_RANGE;
    for(var i = 0; i < data.buildings.size && i < 3000; i++){
        var b = data.buildings.get(i);
        if(b == null || b.block == null) continue;
        var pos = tileOf(b);
        var bx = (pos.x + 0.5) * 8 - vx, by = (pos.y + 0.5) * 8 - vy;
        if(bx * bx + by * by > ownRange) continue;
        own.push({b: b.block.name, x: pos.x, y: pos.y, rot: b.rotation});
    }
    out.own = own;

    var radius = Math.ceil(VISION_RANGE / 8);
    var cx = Math.floor(vx / 8), cy = Math.floor(vy / 8);
    var minX = Math.max(0, cx - radius), maxX = Math.min(Vars.world.width() - 1, cx + radius);
    var minY = Math.max(0, cy - radius), maxY = Math.min(Vars.world.height() - 1, cy + radius);

    var ores = [];
    var blocks = [];

    for(var x = minX; x <= maxX; x++){
        for(var y = minY; y <= maxY; y++){
            var tile = Vars.world.tile(x, y);
            if(tile == null) continue;

            var overlay = tile.overlay();
            if(overlay != null && overlay.itemDrop != null){
                ores.push({i: overlay.itemDrop.name, x: x, y: y});
            }

            var block = tile.block();
            if(block != null && block.name !== "air"){
                var build = tile.build;
                blocks.push({
                    b: block.name,
                    x: x,
                    y: y,
                    team: build != null ? build.team.id : -1,
                    rot: build != null ? build.rotation : 0
                });
            }
        }
    }

    out.ores = ores;
    out.blocks = blocks;

    // enemies inside the ship's vision
    var enemies = [];
    var seenTeams = {};
    var datas = [];
    var active = st.teams.active;
    for(var i = 0; i < active.size; i++){
        var d = active.get(i);
        if(d != null && !seenTeams[d.team.id]){ seenTeams[d.team.id] = true; datas.push(d); }
    }
    var present = st.teams.present;
    for(var i = 0; i < present.size; i++){
        var d = present.get(i);
        if(d != null && !seenTeams[d.team.id]){ seenTeams[d.team.id] = true; datas.push(d); }
    }

    var visionSq = VISION_RANGE * VISION_RANGE;
    for(var i = 0; i < datas.length; i++){
        var td = datas[i];
        if(td.team.id === team.id) continue;
        for(var j = 0; j < td.units.size; j++){
            var u = td.units.get(j);
            if(u == null || u.health <= 0) continue;
            var ux = u.x - vx, uy = u.y - vy;
            if(ux * ux + uy * uy > visionSq) continue;
            enemies.push({
                t: u.type.name,
                x: u.x,
                y: u.y,
                h: u.health,
                team: u.team.id,
                player: u.isPlayer()
            });
        }
    }
    enemies.sort(function(a, b){
        var da = (a.x - vx) * (a.x - vx) + (a.y - vy) * (a.y - vy);
        var db = (b.x - vx) * (b.x - vx) + (b.y - vy) * (b.y - vy);
        return da - db;
    });
    out.enemies = enemies.slice(0, 80);

    return out;
}

// ---------------------------------------------------------------- actions

function blockCost(name){
    var block = findBlock(name);
    if(block == null) throw new Error("unknown block: " + name);
    var req = block.requirements;
    var out = [];
    if(req == null) return out;
    for(var i = 0; i < req.length; i++) out.push([req[i].item.name, req[i].amount]);
    return out;
}

function payFor(name){
    var core = coreBuild();
    if(core == null) throw new Error("no core");
    var cost = blockCost(name);
    for(var i = 0; i < cost.length; i++){
        var item = findItem(cost[i][0]);
        if(core.items.get(item) < cost[i][1]){
            throw new Error("insufficient " + cost[i][0] + " (need " + cost[i][1] + ", have " + core.items.get(item) + ")");
        }
    }
    for(var i = 0; i < cost.length; i++){
        core.items.remove(findItem(cost[i][0]), cost[i][1]);
    }
    return true;
}

function placeRaw(name, tx, ty, rot, paid){
    var block = findBlock(name);
    if(block == null) throw new Error("unknown block: " + name);

    tx = tx | 0; ty = ty | 0;
    var tile = Vars.world.tile(tx, ty);
    if(tile == null) throw new Error("tile out of bounds: " + tx + "," + ty);

    var existing = tile.block();
    if(existing != null && existing.name !== "air") throw new Error("tile occupied: " + tx + "," + ty + " by " + existing.name);

    if(shipUnit != null && shipUnit.isAdded()){
        var dx = tile.worldx() - shipUnit.x, dy = tile.worldy() - shipUnit.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if(dist > shipType.buildRange + 8) throw new Error("out of ship build range: " + Math.round(dist));
    }

    if(paid) payFor(name);
    tile.setBlock(block, gameTeam(), rot | 0);
    return true;
}

function placePaid(name, tx, ty, rot){
    return placeRaw(name, tx, ty, rot, true);
}

function placeFree(name, tx, ty, rot){
    return placeRaw(name, tx, ty, rot, false);
}

function breakAt(tx, ty){
    var tile = Vars.world.tile(tx | 0, ty | 0);
    if(tile == null) throw new Error("tile out of bounds");
    tile.setAir();
    return true;
}

function shipMove(x, y){
    if(steer.target == null) steer.target = new Vec2();
    steer.target.set(x, y);
    return true;
}

function shipHold(){
    steer.target = null;
    return true;
}

function setPaused(paused){
    if(!Vars.state.isGame()) return false;
    var cls = java.lang.Class.forName("mindustry.core.GameState$State");
    var value = java.lang.Enum.valueOf(cls, paused ? "paused" : "playing");
    Vars.state.set(value);
    return true;
}

function countFiles(dir){
    var total = 0;
    var files = dir.list();
    for(var i = 0; i < files.length; i++){
        if(files[i].isDirectory()) total += countFiles(files[i]);
        else total++;
    }
    return total;
}

function copyTree(src, dst){
    dst.mkdirs();
    var files = src.list();
    for(var i = 0; i < files.length; i++){
        var child = files[i];
        var target = dst.child(child.name());
        if(child.isDirectory()) copyTree(child, target);
        else child.copyTo(target);
    }
}

function backupSaves(){
    var src = Vars.saveDirectory;
    var dst = Vars.dataDirectory.child("saves-ai-backup");
    if(src == null || !src.exists()) return {path: dst.absolutePath(), files: 0, error: "no saves directory"};
    if(dst.exists()) dst.deleteDirectory();
    copyTree(src, dst);
    return {path: dst.absolutePath(), files: countFiles(dst)};
}

// ---------------------------------------------------------------- dispatch

function dispatch(msg){
    switch(msg.op){
        case "ping":
            return {pong: true, host: activeHost, time: Time.time};

        case "snapshot":
            return snapshot();

        case "eval":
            return normalize(eval(msg.code == null ? "" : "" + msg.code));

        case "pause":
            return setPaused(!!msg.paused);

        case "chat":
            try{ Vars.ui.chatfrag.addMessage(msg.text == null ? "" : "" + msg.text); }catch(e){}
            return true;

        case "backup":
            return backupSaves();

        case "spawn_ship":
            shipRespawnAt = 0;
            return spawnShip() != null;

        case "ship_move":
            return shipMove(msg.x, msg.y);

        case "ship_hold":
            return shipHold();

        case "ship":
            return {
                alive: shipUnit != null && shipUnit.isAdded(),
                x: shipUnit != null ? shipUnit.x : null,
                y: shipUnit != null ? shipUnit.y : null
            };

        default:
            throw new Error("unknown op: " + msg.op);
    }
}

function safeStringify(value){
    try{
        var text = JSON.stringify(value);
        return text == null ? null : text;
    }catch(e){
        return null;
    }
}

function handleLine(line){
    var msg;
    try{
        msg = JSON.parse(line);
    }catch(e){
        return '{"ok":false,"error":"bad json"}';
    }
    if(msg == null || typeof msg !== "object"){
        return '{"ok":false,"error":"bad request"}';
    }

    var id = msg.id == null ? null : msg.id;
    var res = {id: id};
    try{
        res.data = onMain(function(){ return dispatch(msg); }, 8000);
        res.ok = true;
    }catch(err){
        res.ok = false;
        res.error = "" + (err && err.message ? err.message : err);
    }

    var text = safeStringify(res);
    if(text != null) return text;

    // never let one unserializable result kill the session
    print("[ai-ship] result for op '" + msg.op + "' was not serializable");
    text = safeStringify({id: id, ok: false, error: "result not serializable"});
    return text != null ? text : '{"ok":false,"error":"result not serializable"}';
}

// ---------------------------------------------------------------- socket loop

function connectLoop(){
    while(running){
        var sock = null;
        var host = null;

        for(var i = 0; i < HOSTS.length && sock == null; i++){
            var candidate = null;
            try{
                candidate = new Packages.java.net.Socket();
                candidate.connect(new Packages.java.net.InetSocketAddress(HOSTS[i], PORT), 2000);
                candidate.setTcpNoDelay(true);
                candidate.setKeepAlive(true);
                sock = candidate;
                host = HOSTS[i];
            }catch(e){
                if(candidate != null){
                    try{ candidate.close(); }catch(ignore){}
                }
            }
        }

        if(sock != null){
            activeHost = host;
            connected = true;
            announcedFailure = false;

            try{
                var reader = new Packages.java.io.BufferedReader(
                    new Packages.java.io.InputStreamReader(sock.getInputStream(), "UTF-8"));
                var writer = new Packages.java.io.PrintWriter(
                    new Packages.java.io.OutputStreamWriter(sock.getOutputStream(), "UTF-8"), true);

                writer.println(JSON.stringify({op: "hello", mod: "ai-ship", version: MOD_VERSION, host: host, platform: "android"}));
                say("connected to agent via " + host);

                var line;
                while(running && (line = reader.readLine()) != null){
                    writer.println(handleLine(line));
                }
            }catch(err){
                print("[ai-ship] session error: " + err);
            }finally{
                connected = false;
                activeHost = null;
                try{ sock.close(); }catch(e){}
                say("agent disconnected");
            }
        }else if(!announcedFailure){
            announcedFailure = true;
            print("[ai-ship] no agent listening; retrying quietly");
        }

        try{ Thread.sleep(RECONNECT_MS); }catch(e){}
    }
}

function startBridge(){
    if(started) return;
    started = true;
    running = true;

    var thread = new java.lang.Thread(run(connectLoop));
    thread.setDaemon(true);
    thread.start();
}

// ---------------------------------------------------------------- events

Events.on(ClientLoadEvent, function(){
    try{ startBridge(); }catch(err){ noteError("startBridge", err); }
});

Events.on(WorldLoadEvent, function(){
    try{
        steer.target = null;
        shipUnit = null;
        shipRespawnAt = 0;
        // ensureShip picks up a saved ship or spawns one ~2 seconds after the world loads
        shipEnabled = sectorIsGroundZero();
    }catch(err){
        noteError("worldLoad", err);
    }
});

Events.on(UnitDestroyEvent, function(event){
    try{
        if(shipUnit != null && event.unit != null && event.unit.id === shipUnit.id){
            shipUnit = null;
            shipRespawnAt = Time.time + RESPAWN_SECONDS;
            say("ship destroyed — rebuilding");
        }
    }catch(err){
        noteError("unitDestroy", err);
    }
});

Events.run(Trigger.update, function(){
    ensureShip();
});

Events.on(DisposeEvent, function(){
    running = false;
    connected = false;
});

print("[ai-ship] bridge loaded v" + MOD_VERSION);
