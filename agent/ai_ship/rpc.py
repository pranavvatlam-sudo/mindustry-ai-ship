"""TCP JSON-lines bridge to the AI Ship Mindustry mod.

The mod connects out to us (from inside the Android VM), so this object owns a
server socket and a single active session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("ai_ship.rpc")


class BridgeError(RuntimeError):
    """Raised when a request fails or the link is down."""


@dataclass
class Session:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    hello: dict[str, Any]
    peer: str


class Bridge:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: asyncio.base_events.Server | None = None
        self._session: Session | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self.connected = asyncio.Event()
        self.disconnects = 0

    # ------------------------------------------------------------------ server

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        log.info("bridge listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(BridgeError("bridge stopped"))
        self._pending.clear()

        if self._session is not None:
            self._session.writer.close()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def wait_connected(self, timeout: float = 120.0) -> Session:
        await asyncio.wait_for(self.connected.wait(), timeout)
        assert self._session is not None
        return self._session

    @property
    def session(self) -> Session | None:
        return self._session

    # ------------------------------------------------------------------ session

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = "%s:%s" % writer.get_extra_info("peername")[:2]
        hello: dict[str, Any] = {}

        log.info("mod connected from %s", peer)

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("dropping malformed line: %r", line[:200])
                    continue

                if msg.get("op") == "hello":
                    hello = msg
                    self._session = Session(reader=reader, writer=writer, hello=hello, peer=peer)
                    self.connected.set()
                    log.info(
                        "hello: mod=%s version=%s host=%s platform=%s",
                        hello.get("mod"),
                        hello.get("version"),
                        hello.get("host"),
                        hello.get("platform"),
                    )
                    continue

                rid = msg.get("id")
                fut = self._pending.pop(rid, None) if rid is not None else None
                if fut is not None and not fut.done():
                    fut.set_result(msg)
                else:
                    log.debug("unmatched message: %s", msg)
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            log.info("session ended: %s", exc)
        finally:
            if self._session is not None and self._session.reader is reader:
                self._session = None
                self.connected.clear()
                self.disconnects += 1

            for rid, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_exception(BridgeError("connection lost"))
                self._pending.pop(rid, None)

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            log.info("mod disconnected (%s)", peer)

    # ------------------------------------------------------------------ requests

    async def request(self, op: str, timeout: float = 10.0, **kwargs: Any) -> Any:
        session = self._session
        if session is None:
            raise BridgeError("not connected")

        loop = asyncio.get_running_loop()
        rid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut

        payload = {"id": rid, "op": op}
        payload.update(kwargs)
        session.writer.write((json.dumps(payload) + "\n").encode("utf-8"))

        try:
            await session.writer.drain()
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeError(f"{op} timed out after {timeout}s") from exc
        finally:
            self._pending.pop(rid, None)

        if not msg.get("ok"):
            raise BridgeError(msg.get("error", "unknown mod error"))
        return msg.get("data")

    async def ping(self, timeout: float = 5.0) -> dict:
        return await self.request("ping", timeout=timeout)

    async def snapshot(self, timeout: float = 15.0) -> dict:
        return await self.request("snapshot", timeout=timeout)

    async def eval(self, code: str, timeout: float = 10.0) -> Any:
        return await self.request("eval", code=code, timeout=timeout)

    async def chat(self, text: str) -> Any:
        return await self.request("chat", text=text)
