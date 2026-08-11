"""WebSocket front-end for streaming SenseVoice recognition.

Run it with::

    python -m streaming.ws_server --model-dir iic/SenseVoiceSmall --port 8000

This module is deliberately independent of ``api.py``: that server is a
request/response FastAPI wrapper around ``SenseVoiceSmall.inference`` and owns
its own model instance, whereas this one drives the incremental pipeline in
:mod:`streaming`.  Neither imports the other.

Protocol
--------

The endpoint is any path (``ws://host:port/``).  The connection is symmetric
and framed by the WebSocket layer itself - there is no length prefix or
handshake to perform.

*Client -> server, binary frames*
    Raw PCM: mono, ``config.sample_rate`` Hz (16 kHz), little-endian, either
    ``float32`` in ``[-1, 1]`` (the default) or ``int16``.  Frames may be any
    length; they do not have to align with the model's chunk size.  Because
    both formats are just bytes, the encoding is **declared, not sniffed**:
    send ``{"type": "config", "format": "int16"}`` before the first audio
    frame, or connect to ``ws://host:port/?format=int16``.  A frame whose
    length is not a whole number of samples is rejected with an ``error``.

*Client -> server, text frames* (JSON object, or the bare command word)
    ``{"type": "config", "format": "float32"|"int16"}``
        Set the PCM encoding of the binary frames that *follow*.  It may be
        sent at any point in the connection, not only before the first audio
        frame: each binary frame is decoded with the encoding in force when it
        arrives, so a ``config`` never re-interprets audio already received.
        Answered with ``{"type": "config", "format": ...}``; an unknown
        encoding is rejected with an ``error`` and leaves the current one in
        place.
    ``{"type": "eof"}``
        Close the current segment: the pipeline is flushed with
        ``is_final=True``, so an utterance in progress yields its ``final``.
        The server answers with ``{"type": "eof"}`` once the flush is done.
        The connection stays open and a new utterance may follow.
    ``{"type": "reset"}``
        Discard all stream state without emitting a final.  Answered with
        ``{"type": "reset"}``.

*Server -> client, text frames* (JSON object)
    ``{"type": "ready", ...}``
        Sent once on connect, carrying the negotiated audio parameters.
    ``{"type": "partial", "text": ...}``
        In-progress transcript of the current utterance.  Supersedes the
        previous ``partial``; it is not an increment to append.
    ``{"type": "final", "text": ...}``
        Settled transcript of one utterance.  No further message will revise
        it.
    ``{"type": "error", "message": ...}``
        A malformed frame or a failed inference.  Recoverable errors leave the
        connection open.

    ``partial`` and ``final`` also carry ``raw_text`` (the transcript with
    SenseVoice's ``<|...|>`` rich tags still in place) and ``start_ms`` /
    ``end_ms`` (the utterance's span on the stream clock).  Clients that only
    need the transcript can read ``type`` and ``text`` alone.

Model sharing and thread safety
-------------------------------

Three independent constraints shape the design.

*Recognisers are reused, not shared.*  ``StreamingSenseVoice`` is ~900 MB
resident, and an instance carries the encoder history and cache of *one* audio
stream; the class exposes no way to swap that state out, so two live streams
cannot take turns on a single instance without interleaving their histories
into nonsense.  Concurrency therefore costs memory: the structure is a **pool
of recognisers** (:class:`RecognizerPool`) whose size is the real cap
(``--max-sessions``, default 1), and **each slot holds its own copy of the
weights** - ``--max-sessions N`` budgets for roughly *900 MB x N*.  What a slot
saves is the *sequential* case: a connection borrows an instance for its
lifetime and returns it on disconnect, so the next connection reuses the loaded
weights instead of paying for a fresh load.  Nothing is shared between
*simultaneous* connections.  A connection that arrives with the pool exhausted
is rejected with an explicit ``error`` frame instead of being served a
corrupted transcript.  Slots are filled lazily, so an idle server holds no
weights at all.

*The VAD, in contrast, really is shared.*  ``fsmn-vad`` keeps no state of its
own between calls - the stream state lives in the ``cache`` dict each
:class:`~streaming.vad_gate.VadGate` owns - so a single ``funasr.AutoModel``
serves every connection.  :class:`StreamingServer` loads exactly one, lazily on
the first connection and behind an :class:`asyncio.Lock` so a burst of
connections cannot start several loads, and injects it into each connection's
gate.

*Inference is serialised.*  The encoder is CPU-bound and already configured for
``config.num_threads`` threads; running two forward passes at once would only
oversubscribe the cores and push both past real time.  All inference therefore
runs on a **single-worker** ``ThreadPoolExecutor``.  Using a one-thread
executor rather than a lock is what keeps the asyncio loop responsive: the
handler ``await``\\ s the future, so the loop keeps reading sockets and
flushing results while the worker computes, and the executor's queue *is* the
mutual exclusion - there is no lock to forget to take.

Model *loading* gets a **second** executor.  Filling a pool slot takes seconds
and is not inference; queueing it behind the same single worker would stall
every established connection's audio for the duration (head-of-line blocking on
the one thread that matters).  The loader executor is separate so a new
connection warming up never delays a stream already running, and both are shut
down by :meth:`StreamingServer.close`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .config import StreamingConfig
from .session import DEFAULT_LOOK_BACK_SEC, StreamingSession
from .vad_gate import VadGate

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from websockets.asyncio.server import ServerConnection

    from .streaming_model import StreamingResult, StreamingSenseVoice

__all__ = [
    "PCM_FORMATS",
    "RecognizerPool",
    "StreamingServer",
    "build_parser",
    "decode_pcm",
    "main",
]

LOGGER = logging.getLogger(__name__)

#: Accepted binary encodings, mapped to their numpy dtype.  Both are
#: little-endian so a browser's ``AudioContext`` output can be sent as-is.
PCM_FORMATS: Dict[str, np.dtype] = {
    "float32": np.dtype("<f4"),
    "int16": np.dtype("<i2"),
}

#: Scale that maps full-scale ``int16`` onto ``[-1, 1]``.
_INT16_SCALE = 1.0 / 32768.0


def decode_pcm(payload: bytes, pcm_format: str) -> np.ndarray:
    """Decode a binary audio frame into mono ``float32`` samples.

    Args:
        payload: Raw little-endian PCM bytes.
        pcm_format: One of the keys of :data:`PCM_FORMATS`.

    Returns:
        A 1-D ``float32`` array in ``[-1, 1]``.  ``int16`` input is scaled;
        ``float32`` input is passed through unchanged (and copied, so the
        caller's buffer can be reused).

    Raises:
        ValueError: If ``pcm_format`` is unknown, or if ``payload`` does not
            hold a whole number of samples.
    """
    try:
        dtype = PCM_FORMATS[pcm_format]
    except KeyError:
        raise ValueError(
            f"unsupported format {pcm_format!r}; expected one of "
            f"{sorted(PCM_FORMATS)}"
        ) from None
    if len(payload) % dtype.itemsize:
        raise ValueError(
            f"{len(payload)} bytes is not a whole number of {pcm_format} "
            f"samples ({dtype.itemsize} bytes each)"
        )
    samples = np.frombuffer(payload, dtype=dtype)
    if dtype.kind == "i":
        return samples.astype(np.float32) * _INT16_SCALE
    return samples.astype(np.float32)


def _result_payload(result: "StreamingResult") -> Dict[str, Any]:
    """Render a recogniser result as the JSON object sent to clients.

    ``type`` and ``text`` are always present and are all a minimal client
    needs; the remaining fields are passed through from the dataclass.
    """
    fields: Dict[str, Any] = (
        dict(asdict(result)) if is_dataclass(result) else {}
    )
    fields.update(
        {
            "type": result.type,
            "text": result.text,
            "raw_text": getattr(result, "raw_text", ""),
            "start_ms": getattr(result, "start_ms", 0.0),
            "end_ms": getattr(result, "end_ms", 0.0),
        }
    )
    return fields


def _build_recognizer(
    model_dir: str, config: StreamingConfig
) -> "StreamingSenseVoice":
    """Load one :class:`StreamingSenseVoice`, importing torch only now.

    Args:
        model_dir: Model id or local directory.
        config: Streaming configuration handed to the instance.

    Returns:
        A freshly loaded recogniser (~900 MB resident).
    """
    from .streaming_model import StreamingSenseVoice

    return StreamingSenseVoice(model_dir, config)


def _build_vad_model(config: StreamingConfig) -> Any:
    """Load the shared FunASR VAD described by ``config``.

    ``funasr`` is imported here rather than at module scope so this module
    stays importable on a checkout without the ML dependencies.  The result is
    stateless across calls - fsmn-vad keeps its per-stream state in the
    ``cache`` dict its caller owns - which is what makes one instance safe to
    hand to every connection.

    Args:
        config: Streaming configuration; supplies ``vad_model`` and ``device``.

    Returns:
        A ``funasr.AutoModel`` exposing ``generate(input=..., cache=..., ...)``.
    """
    from funasr import AutoModel

    return AutoModel(
        model=config.vad_model,
        disable_update=True,
        disable_pbar=True,
        device=config.device,
    )


class RecognizerPool:
    """Bounded, lazily filled pool of :class:`StreamingSenseVoice` instances.

    Each slot is created on first use and then reused forever, so the process
    never holds more than ``size`` copies of the weights and a server that is
    never connected to loads nothing.  Borrowing is exclusive: a slot is out of
    the pool for as long as a connection holds it, which is what keeps two
    streams from sharing one instance's encoder history.  Slots are *not*
    lighter than a standalone recogniser: every filled slot is its own ~900 MB
    instance, so ``size`` multiplies the resident memory.

    Args:
        model_dir: Model id or local directory passed to
            :class:`StreamingSenseVoice`.
        config: Streaming configuration handed to every instance.
        size: Maximum number of concurrent recognisers, i.e. of concurrent
            connections.  Sized by memory, not by CPU - inference is
            serialised elsewhere.
        factory: Builds one recogniser from ``(model_dir, config)``.  Defaults
            to constructing a :class:`StreamingSenseVoice`, imported only when
            a slot is actually filled so that this module stays importable
            without torch; tests inject a stub instead.

    Raises:
        ValueError: If ``size`` is smaller than 1.
    """

    def __init__(
        self,
        model_dir: str,
        config: StreamingConfig,
        size: int = 1,
        factory: Optional[
            Callable[[str, StreamingConfig], "StreamingSenseVoice"]
        ] = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1, got {size}")
        self.model_dir = model_dir
        self.config = config
        self.size = size
        self.factory = factory if factory is not None else _build_recognizer
        self._slots: "queue.Queue[Optional[StreamingSenseVoice]]" = queue.Queue()
        for _ in range(size):
            self._slots.put(None)

    def acquire(self) -> "StreamingSenseVoice":
        """Take a recogniser out of the pool, building it if the slot is empty.

        Returns:
            A recogniser owned exclusively by the caller until
            :meth:`release`.  Its stream state is reset before it is handed
            over, so a reused instance carries nothing from the previous
            connection.

        Raises:
            RuntimeError: If every slot is already borrowed.
        """
        try:
            recognizer = self._slots.get_nowait()
        except queue.Empty:
            raise RuntimeError(
                f"all {self.size} recogniser slots are in use; "
                f"raise --max-sessions to serve more concurrent connections"
            ) from None
        try:
            if recognizer is None:
                LOGGER.info("loading recogniser from %s", self.model_dir)
                recognizer = self.factory(self.model_dir, self.config)
            else:
                recognizer.reset()
        except BaseException:
            # Never lose a slot: a failed load must leave the pool able to
            # try again on the next connection.
            self._slots.put(None)
            raise
        return recognizer

    def release(self, recognizer: "StreamingSenseVoice") -> None:
        """Return a recogniser to the pool for the next connection."""
        self._slots.put(recognizer)


class StreamingServer:
    """Serve streaming recognition over WebSocket.

    Args:
        model_dir: Model id or local directory for the recogniser.
        config: Streaming configuration shared by all connections.
        max_sessions: Pool size, i.e. the number of connections that may
            recognise at the same time.  Each one costs a full copy of the
            weights (~900 MB).
        look_back_sec: Seconds of audio each session may rewind into when a
            segment opens; see :class:`~streaming.session.StreamingSession`.
        recognizer_factory: Override for how a pool slot is filled; see
            :class:`RecognizerPool`.  Injected by tests to avoid loading
            weights.
        vad_factory: Override for how the shared VAD is loaded, called once
            with the configuration.  Injected by tests to avoid FunASR.
    """

    def __init__(
        self,
        model_dir: str,
        config: Optional[StreamingConfig] = None,
        max_sessions: int = 1,
        look_back_sec: float = DEFAULT_LOOK_BACK_SEC,
        recognizer_factory: Optional[
            Callable[[str, StreamingConfig], "StreamingSenseVoice"]
        ] = None,
        vad_factory: Optional[Callable[[StreamingConfig], Any]] = None,
    ) -> None:
        self.config = config if config is not None else StreamingConfig()
        self.config.validate()
        self.pool = RecognizerPool(
            model_dir,
            self.config,
            size=max_sessions,
            factory=recognizer_factory,
        )
        self.look_back_sec = look_back_sec
        # One worker: all inference is serialised, and the asyncio loop is
        # never the thing doing it.  See the module docstring.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sensevoice-infer"
        )
        # Loading weights is slow and is *not* inference: it gets its own
        # thread so a connection warming up never sits in front of the audio of
        # a connection already running.
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sensevoice-load"
        )
        self._vad_factory = (
            vad_factory if vad_factory is not None else _build_vad_model
        )
        self._vad_model: Optional[Any] = None
        self._vad_lock: Optional[asyncio.Lock] = None

    # --------------------------------------------------------------- lifetime

    async def serve_forever(self, host: str, port: int) -> None:
        """Listen on ``host:port`` until the process is asked to stop.

        Stops cleanly on ``SIGINT``/``SIGTERM`` where the platform supports
        signal handlers on the loop, and on ``KeyboardInterrupt`` otherwise.
        """
        from websockets.asyncio.server import serve

        loop = asyncio.get_running_loop()
        stop: "asyncio.Future[None]" = loop.create_future()

        def request_stop() -> None:
            """Wake ``serve_forever``; repeat signals are harmless."""
            if not stop.done():
                stop.set_result(None)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except (NotImplementedError, RuntimeError):  # pragma: no cover
                pass

        async with serve(self.handle, host, port):
            LOGGER.info("listening on ws://%s:%d", host, port)
            try:
                await stop
            except KeyboardInterrupt:  # pragma: no cover - console only
                pass
        LOGGER.info("server stopped")

    def close(self) -> None:
        """Shut both workers down, waiting for the work in flight.

        The inference worker is drained first so a pass already running is not
        abandoned; the loader is drained after, since a load in flight belongs
        to a connection that is going away anyway but still holds a pool slot.
        """
        self._executor.shutdown(wait=True)
        self._load_executor.shutdown(wait=True)

    # ------------------------------------------------------------------ models

    async def _get_vad_model(self) -> Any:
        """Return the process-wide VAD, loading it on first use.

        fsmn-vad is stateless between calls, so one instance serves every
        connection and the load is worth doing exactly once.  The load runs on
        the loader executor (never on the event loop, which a synchronous
        ``AutoModel(...)`` would block for seconds) and behind a lock, so a
        burst of connections arriving together waits on one load rather than
        starting several.

        Returns:
            The shared VAD model, ready to be injected into a
            :class:`~streaming.vad_gate.VadGate`.
        """
        if self._vad_model is not None:
            return self._vad_model
        if self._vad_lock is None:
            # Created here rather than in ``__init__`` so the lock binds to the
            # loop that actually serves; safe without a lock of its own because
            # there is no ``await`` between the test and the assignment.
            self._vad_lock = asyncio.Lock()
        async with self._vad_lock:
            if self._vad_model is None:
                LOGGER.info("loading shared VAD %s", self.config.vad_model)
                loop = asyncio.get_running_loop()
                self._vad_model = await loop.run_in_executor(
                    self._load_executor, self._vad_factory, self.config
                )
        return self._vad_model

    # --------------------------------------------------------------- handling

    async def handle(self, websocket: "ServerConnection") -> None:
        """Serve one client connection.

        Borrows a recogniser for the whole connection, wires it to a fresh
        :class:`~streaming.session.StreamingSession` over the shared VAD, and
        pumps frames until the client goes away.  The recogniser is always
        returned to the pool, including on error.
        """
        from websockets.exceptions import ConnectionClosed

        try:
            recognizer = await asyncio.get_running_loop().run_in_executor(
                self._load_executor, self.pool.acquire
            )
        except RuntimeError as exc:
            await self._send(websocket, {"type": "error", "message": str(exc)})
            await websocket.close(code=1013, reason="server busy")
            return
        except Exception as exc:  # pragma: no cover - model load failure
            LOGGER.exception("could not create a recogniser")
            await self._send(
                websocket,
                {"type": "error", "message": f"model unavailable: {exc}"},
            )
            await websocket.close(code=1011, reason="model unavailable")
            return

        try:
            try:
                vad_model = await self._get_vad_model()
            except Exception as exc:  # pragma: no cover - VAD load failure
                LOGGER.exception("could not load the VAD")
                await self._send(
                    websocket,
                    {"type": "error", "message": f"vad unavailable: {exc}"},
                )
                await websocket.close(code=1011, reason="vad unavailable")
                return
            session = StreamingSession(
                model=recognizer,
                vad=VadGate(self.config, vad_model=vad_model),
                config=self.config,
                look_back_sec=self.look_back_sec,
            )
            pcm_format = _format_from_path(getattr(websocket, "request", None))
            await self._send(
                websocket,
                {
                    "type": "ready",
                    "sample_rate": self.config.sample_rate,
                    "format": pcm_format,
                    "chunk_ms": self.config.chunk_ms,
                },
            )
            await self._pump(websocket, session, pcm_format)
        except ConnectionClosed:
            LOGGER.debug("client disconnected")
        finally:
            self.pool.release(recognizer)

    async def _pump(
        self,
        websocket: "ServerConnection",
        session: StreamingSession,
        pcm_format: str,
    ) -> None:
        """Read frames from one client until it disconnects."""
        async for message in websocket:
            if isinstance(message, (bytes, bytearray, memoryview)):
                try:
                    samples = decode_pcm(bytes(message), pcm_format)
                except ValueError as exc:
                    await self._send(
                        websocket, {"type": "error", "message": str(exc)}
                    )
                    continue
                await self._recognize(websocket, session, samples, False)
                continue

            command, payload = _parse_command(str(message))
            if command == "eof":
                await self._recognize(
                    websocket, session, _EMPTY_AUDIO, True
                )
                await self._send(websocket, {"type": "eof"})
            elif command == "reset":
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, session.reset
                )
                await self._send(websocket, {"type": "reset"})
            elif command == "config":
                new_format = str(payload.get("format", pcm_format))
                if new_format not in PCM_FORMATS:
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "message": (
                                f"unsupported format {new_format!r}; expected "
                                f"one of {sorted(PCM_FORMATS)}"
                            ),
                        },
                    )
                else:
                    pcm_format = new_format
                    await self._send(
                        websocket, {"type": "config", "format": pcm_format}
                    )
            else:
                await self._send(
                    websocket,
                    {
                        "type": "error",
                        "message": f"unknown command {command!r}",
                    },
                )

    async def _recognize(
        self,
        websocket: "ServerConnection",
        session: StreamingSession,
        samples: np.ndarray,
        is_final: bool,
    ) -> None:
        """Run one block through the pipeline and forward what it produced."""
        loop = asyncio.get_running_loop()
        try:
            results: List["StreamingResult"] = await loop.run_in_executor(
                self._executor, session.push_audio, samples, is_final
            )
        except Exception as exc:
            LOGGER.exception("inference failed")
            await self._send(
                websocket,
                {"type": "error", "message": f"inference failed: {exc}"},
            )
            return
        for result in results:
            await self._send(websocket, _result_payload(result))

    @staticmethod
    async def _send(
        websocket: "ServerConnection", payload: Dict[str, Any]
    ) -> None:
        """Send one JSON text frame, ignoring an already-closed connection."""
        from websockets.exceptions import ConnectionClosed

        try:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed:
            LOGGER.debug("dropped %s frame: connection closed", payload["type"])


#: Reused for the zero-length flush that ``eof`` performs.
_EMPTY_AUDIO = np.zeros(0, dtype=np.float32)


def _parse_command(message: str) -> Tuple[str, Dict[str, Any]]:
    """Interpret a text frame as ``(command, payload)``.

    Accepts a JSON object with a ``type`` key, and also a bare command word
    (``eof``) so a client can be written with ``ws.send("eof")``.
    """
    text = message.strip()
    if text[:1] != "{":
        return text.lower(), {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "", {}
    if not isinstance(payload, dict):
        return "", {}
    return str(payload.get("type", "")).lower(), payload


def _format_from_path(request: Any) -> str:
    """Read the ``format`` query parameter of the handshake, if any.

    Falls back to ``"float32"`` when the request carries no usable value, so a
    client that just connects to ``ws://host:port/`` gets the common case.
    """
    path = getattr(request, "path", None)
    if not path:
        return "float32"
    values = parse_qs(urlsplit(path).query).get("format")
    if not values:
        return "float32"
    candidate = values[0].lower()
    return candidate if candidate in PCM_FORMATS else "float32"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``python -m streaming.ws_server``."""
    parser = argparse.ArgumentParser(
        description="WebSocket server for streaming SenseVoice recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = StreamingConfig()
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--model-dir",
        default="iic/SenseVoiceSmall",
        help="model id or local model directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        help="encoder frames per inference (1 frame = 60 ms)",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=defaults.max_history,
        help="encoder frames of context kept across chunks",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help=(
            "torch device the models run on (cpu, cuda:0, ...); this is not "
            "an audio device"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help=(
            "connections served at once; each holds its own copy of the "
            "weights, so memory is about 900 MB per session"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run the server until interrupted.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit status: ``0`` on a clean shutdown, ``2`` when the
        configuration is rejected.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = StreamingConfig(
        chunk_size=args.chunk_size,
        max_history=args.max_history,
        device=args.device,
    )
    try:
        config.validate()
        server = StreamingServer(
            model_dir=args.model_dir,
            config=config,
            max_sessions=args.max_sessions,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    try:
        asyncio.run(server.serve_forever(args.host, args.port))
    except KeyboardInterrupt:  # pragma: no cover - console only
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
