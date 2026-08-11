"""Unit tests for streaming/ws_server.py.

Everything exercised here is either a pure function (:func:`decode_pcm`,
:func:`_parse_command`, :func:`_format_from_path`) or a piece of plumbing whose
model construction is injectable (:class:`RecognizerPool`,
:class:`StreamingServer`'s shared VAD), so the suite needs neither torch,
funasr nor websockets - only numpy, which is guarded with ``importorskip`` like
the other ML-adjacent tests in this directory.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from streaming.config import StreamingConfig  # noqa: E402
from streaming.ws_server import (  # noqa: E402
    PCM_FORMATS,
    RecognizerPool,
    StreamingServer,
    _format_from_path,
    _parse_command,
    build_parser,
    decode_pcm,
)


class StubRecognizer:
    """Stands in for ``StreamingSenseVoice`` without loading any weights.

    Args:
        model_dir: Recorded so a test can assert what the pool passed on.
        config: Recorded for the same reason.
    """

    def __init__(self, model_dir: str, config: StreamingConfig) -> None:
        self.model_dir = model_dir
        self.config = config
        self.resets = 0

    def reset(self) -> None:
        """Count the resets the pool performs when a slot is reused."""
        self.resets += 1


class CountingFactory:
    """Recogniser factory that counts the instances it was asked to build."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, StreamingConfig]] = []

    def __call__(
        self, model_dir: str, config: StreamingConfig
    ) -> StubRecognizer:
        """Build one :class:`StubRecognizer` and remember the arguments."""
        self.calls.append((model_dir, config))
        return StubRecognizer(model_dir, config)


# ------------------------------------------------------------------ decode_pcm


def test_decode_pcm_float32_roundtrips_values() -> None:
    """float32 input comes back unchanged, as its own array."""
    source = np.array([0.0, 0.5, -0.25, 1.0], dtype="<f4")
    decoded = decode_pcm(source.tobytes(), "float32")

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, [0.0, 0.5, -0.25, 1.0])


def test_decode_pcm_float32_copies_the_caller_buffer() -> None:
    """The result owns its memory: ``np.frombuffer`` alone would not."""
    payload = np.array([0.25], dtype="<f4").tobytes()
    decoded = decode_pcm(payload, "float32")

    assert decoded.flags.writeable
    assert decoded.base is None or not isinstance(decoded.base, bytes)
    decoded[0] = 1.0  # must not raise, and must not corrupt ``payload``
    assert payload == np.array([0.25], dtype="<f4").tobytes()


def test_decode_pcm_int16_is_scaled_to_unit_range() -> None:
    """int16 is mapped onto [-1, 1) by dividing by 32768."""
    source = np.array([0, 16384, -32768, 32767], dtype="<i2")
    decoded = decode_pcm(source.tobytes(), "int16")

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(
        decoded, [0.0, 0.5, -1.0, 32767 / 32768.0], atol=1e-7
    )


def test_decode_pcm_accepts_an_empty_frame() -> None:
    """A zero-length frame is a whole number of samples: zero of them."""
    for pcm_format in PCM_FORMATS:
        decoded = decode_pcm(b"", pcm_format)
        assert decoded.dtype == np.float32
        assert decoded.size == 0


@pytest.mark.parametrize(
    ("payload", "pcm_format"),
    [
        (b"\x00\x00\x00", "float32"),
        (b"\x00" * 5, "float32"),
        (b"\x00", "int16"),
        (b"\x00" * 3, "int16"),
    ],
)
def test_decode_pcm_rejects_a_partial_sample(
    payload: bytes, pcm_format: str
) -> None:
    """A frame that stops mid-sample is an error, not a silent truncation."""
    with pytest.raises(ValueError, match="whole number"):
        decode_pcm(payload, pcm_format)


def test_decode_pcm_rejects_an_unknown_format() -> None:
    """An encoding outside PCM_FORMATS names the accepted ones."""
    with pytest.raises(ValueError, match="unsupported format"):
        decode_pcm(b"", "float64")


# --------------------------------------------------------------- _parse_command


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ('{"type": "eof"}', "eof"),
        ('{"type": "reset"}', "reset"),
        ('{"type": "config", "format": "int16"}', "config"),
        ('  {"type": "EOF"}  ', "eof"),
    ],
)
def test_parse_command_reads_json_objects(message: str, expected: str) -> None:
    """A JSON object yields its lower-cased ``type``."""
    command, payload = _parse_command(message)

    assert command == expected
    assert payload == json.loads(message.strip())


def test_parse_command_accepts_a_bare_command_word() -> None:
    """``ws.send("eof")`` is a supported shorthand and carries no payload."""
    assert _parse_command("eof") == ("eof", {})
    assert _parse_command("  RESET \n") == ("reset", {})


def test_parse_command_returns_the_config_payload() -> None:
    """The whole object is handed back so ``format`` can be read from it."""
    command, payload = _parse_command('{"type": "config", "format": "int16"}')

    assert command == "config"
    assert payload["format"] == "int16"


@pytest.mark.parametrize(
    "message",
    ['{"type": "eof"', "{}}", '{"type": }', "{"],
)
def test_parse_command_rejects_malformed_json(message: str) -> None:
    """Broken JSON degrades to the empty command, never an exception."""
    assert _parse_command(message) == ("", {})


@pytest.mark.parametrize("message", ["[1, 2]", '"eof"', "null"])
def test_parse_command_rejects_json_that_is_not_an_object(
    message: str,
) -> None:
    """Valid JSON of the wrong shape never yields a payload.

    Only a frame starting with ``{`` is parsed as JSON at all, so these take
    the bare-word path and surface as unknown commands - notably ``"eof"``
    with its quotes is *not* accepted as ``eof``.
    """
    command, payload = _parse_command(message)

    assert payload == {}
    assert command not in {"eof", "reset", "config"}


def test_parse_command_passes_unknown_types_through() -> None:
    """An unknown ``type`` is reported as-is so the caller can complain."""
    assert _parse_command('{"type": "wibble"}')[0] == "wibble"
    assert _parse_command("wibble") == ("wibble", {})
    assert _parse_command('{"format": "int16"}')[0] == ""


def test_parse_command_handles_an_empty_frame() -> None:
    """An empty text frame is an unknown command, not a crash."""
    assert _parse_command("") == ("", {})
    assert _parse_command("   ") == ("", {})


# ------------------------------------------------------------ _format_from_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/", "float32"),
        ("/?format=int16", "int16"),
        ("/?format=INT16", "int16"),
        ("/?format=float32", "float32"),
        ("/?format=float64", "float32"),
        ("/?other=1", "float32"),
    ],
)
def test_format_from_path(path: str, expected: str) -> None:
    """The handshake query selects an encoding, defaulting to float32."""

    class Request:
        """Minimal stand-in for a websockets ``Request``."""

        def __init__(self, path: str) -> None:
            self.path = path

    assert _format_from_path(Request(path)) == expected


def test_format_from_path_without_a_request() -> None:
    """A missing request object falls back to the common case."""
    assert _format_from_path(None) == "float32"


# ------------------------------------------------------------- RecognizerPool


def test_pool_rejects_a_size_below_one() -> None:
    """A pool that can never hand anything out is a configuration error."""
    for size in (0, -1):
        with pytest.raises(ValueError, match="pool size must be >= 1"):
            RecognizerPool("model", StreamingConfig(), size=size)


def test_pool_builds_lazily_and_reuses_the_instance() -> None:
    """Nothing loads until the first acquire; the second reuses the slot."""
    factory = CountingFactory()
    pool = RecognizerPool(
        "model-dir", StreamingConfig(), size=1, factory=factory
    )
    assert factory.calls == []

    first = pool.acquire()
    assert factory.calls == [("model-dir", pool.config)]
    assert first.resets == 0

    pool.release(first)
    second = pool.acquire()

    assert second is first
    assert len(factory.calls) == 1
    assert second.resets == 1, "a reused instance must be reset before reuse"


def test_pool_hands_out_distinct_instances_up_to_its_size() -> None:
    """Borrowing is exclusive: two connections never get the same recogniser."""
    factory = CountingFactory()
    pool = RecognizerPool("model", StreamingConfig(), size=2, factory=factory)

    first = pool.acquire()
    second = pool.acquire()

    assert first is not second
    assert len(factory.calls) == 2


def test_pool_refuses_to_over_subscribe() -> None:
    """An exhausted pool raises rather than sharing a recogniser."""
    pool = RecognizerPool(
        "model", StreamingConfig(), size=1, factory=CountingFactory()
    )
    borrowed = pool.acquire()

    with pytest.raises(RuntimeError, match="all 1 recogniser slots are in use"):
        pool.acquire()

    pool.release(borrowed)
    assert pool.acquire() is borrowed


def test_pool_keeps_the_slot_when_the_load_fails() -> None:
    """A failed load must not permanently shrink the pool."""
    attempts: List[int] = []

    def flaky(model_dir: str, config: StreamingConfig) -> StubRecognizer:
        """Fail the first build, succeed afterwards."""
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("weights not found")
        return StubRecognizer(model_dir, config)

    pool = RecognizerPool("model", StreamingConfig(), size=1, factory=flaky)

    with pytest.raises(OSError):
        pool.acquire()

    recognizer = pool.acquire()
    assert isinstance(recognizer, StubRecognizer)
    assert len(attempts) == 2


# ------------------------------------------------------- shared VAD on server


def _make_server(vad_factory: Any) -> StreamingServer:
    """Build a server whose models are stubs, so no weights are loaded."""
    return StreamingServer(
        model_dir="model",
        config=StreamingConfig(),
        max_sessions=1,
        recognizer_factory=lambda model_dir, config: StubRecognizer(
            model_dir, config
        ),
        vad_factory=vad_factory,
    )


def test_server_loads_the_vad_once_and_shares_it() -> None:
    """Repeated requests for the VAD return the one instance, loaded once."""
    loads: List[StreamingConfig] = []

    def loader(config: StreamingConfig) -> object:
        """Record the load and hand back a unique sentinel."""
        loads.append(config)
        return object()

    server = _make_server(loader)
    try:

        async def scenario() -> Tuple[Any, Any]:
            """Ask for the VAD twice, sequentially."""
            return await server._get_vad_model(), await server._get_vad_model()

        first, second = asyncio.run(scenario())
    finally:
        server.close()

    assert first is second
    assert loads == [server.config]


def test_server_does_not_load_the_vad_twice_under_a_burst() -> None:
    """Connections arriving together wait on one load, not several."""
    started = threading.Event()
    release = threading.Event()
    loads: List[int] = []

    def slow_loader(config: StreamingConfig) -> object:
        """Block until the test lets go, so the callers overlap for sure."""
        loads.append(1)
        started.set()
        release.wait(timeout=5.0)
        return object()

    server = _make_server(slow_loader)
    try:

        async def scenario() -> List[Any]:
            """Five simultaneous connections asking for the VAD at once."""
            waiters = [
                asyncio.create_task(server._get_vad_model()) for _ in range(5)
            ]
            await asyncio.to_thread(started.wait, 5.0)
            release.set()
            return await asyncio.gather(*waiters)

        models = asyncio.run(scenario())
    finally:
        release.set()
        server.close()

    assert loads == [1], "the lock must collapse a burst into a single load"
    assert all(model is models[0] for model in models)


def test_server_does_not_load_the_vad_on_the_event_loop() -> None:
    """The load runs on a worker thread, never on the loop's own thread."""
    loop_thread: List[int] = []
    load_thread: List[int] = []

    def loader(config: StreamingConfig) -> object:
        """Record which thread the load happened on."""
        load_thread.append(threading.get_ident())
        return object()

    server = _make_server(loader)
    try:

        async def scenario() -> None:
            """Note the loop's thread, then trigger the load."""
            loop_thread.append(threading.get_ident())
            await server._get_vad_model()

        asyncio.run(scenario())
    finally:
        server.close()

    assert load_thread and load_thread[0] != loop_thread[0]


def test_server_keeps_loading_and_inference_on_separate_workers() -> None:
    """Head-of-line blocking is avoided by not sharing the one worker."""
    server = _make_server(lambda config: object())
    try:
        assert server._executor is not server._load_executor
    finally:
        server.close()


def test_server_close_shuts_both_executors_down() -> None:
    """``close`` must leave no thread pool running."""
    server = _make_server(lambda config: object())
    server.close()

    with pytest.raises(RuntimeError):
        server._executor.submit(int)
    with pytest.raises(RuntimeError):
        server._load_executor.submit(int)


# ------------------------------------------------------------------------ CLI


def test_parser_defaults_match_the_streaming_config() -> None:
    """The CLI must not drift from :class:`StreamingConfig`'s defaults."""
    defaults = StreamingConfig()
    args = build_parser().parse_args([])

    assert args.chunk_size == defaults.chunk_size
    assert args.max_history == defaults.max_history
    assert args.device == defaults.device
    assert args.max_sessions == 1


def test_device_help_says_it_is_a_torch_device() -> None:
    """``--device`` is the torch device, not the microphone."""
    help_text = build_parser().format_help()

    assert "torch device" in help_text
    assert "not an audio device" in help_text


def test_max_sessions_help_warns_about_memory() -> None:
    """Each session costs a full copy of the weights; say so."""
    assert "900 MB per session" in build_parser().format_help()
