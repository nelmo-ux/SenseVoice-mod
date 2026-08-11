"""Command-line demo of streaming SenseVoice recognition.

Talk into the microphone and watch the transcript settle::

    python demo_streaming_mic.py --model-dir iic/SenseVoiceSmall

In-progress text is rewritten in place on one line; each time the endpointer
closes an utterance the settled text is committed with a newline, so the
terminal ends up holding the transcript and nothing else.  ``Ctrl+C`` stops.

Microphone capture needs the optional ``sounddevice`` package.  Without it the
demo still runs from a file, which is also the reproducible way to compare
runs::

    python demo_streaming_mic.py --wav sample.wav

``--wav`` feeds the file at wall-clock speed, so it exercises the pipeline
under the same timing pressure as live audio; pass ``--no-realtime`` to run it
as fast as the CPU allows.  Any sample rate and channel count is accepted:
audio is downmixed to mono and resampled to the model's 16 kHz.

Two unrelated devices are configurable, and they are named apart:
``--input-device`` selects the *microphone* (a PortAudio index or name), while
``--torch-device`` selects where the *models run* - the same meaning
``--device`` carries in ``streaming/ws_server.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

import numpy as np

from streaming.config import StreamingConfig
from streaming.session import StreamingSession
from streaming.vad_gate import VadGate

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from streaming.streaming_model import StreamingResult

__all__ = ["TranscriptPrinter", "build_parser", "display_width", "load_wav", "main"]

#: Terminal size assumed when the stream is not a terminal (a pipe or a file).
_FALLBACK_TERMINAL_SIZE = (80, 24)

#: Marker appended to a line cut short by the terminal width.  One cell wide.
_ELLIPSIS = "…"

#: Shown when capture is requested but the optional dependency is absent.
_SOUNDDEVICE_HINT = (
    "microphone capture needs the optional 'sounddevice' package, which is "
    "not installed.\n"
    "  install it with:  pip install sounddevice\n"
    "  or run without a microphone:  python demo_streaming_mic.py --wav <file>"
)


def display_width(text: str) -> int:
    """Return how many terminal cells ``text`` occupies.

    CJK transcripts are the normal case here, and a Chinese character occupies
    two cells while ``len()`` counts it as one - so padding computed from
    ``len()`` clears only half of what the previous line drew and leaves its
    tail on screen.

    Args:
        text: Any string, typically one already-rendered transcript line.

    Returns:
        The sum of the per-character widths: ``2`` for East Asian *Wide* and
        *Fullwidth* characters, ``1`` for everything else.  Control characters
        and combining marks are not special-cased; the caller only ever passes
        printable transcript text.
    """
    return sum(_char_width(char) for char in text)


def _char_width(char: str) -> int:
    """Width of a single character in terminal cells (``1`` or ``2``)."""
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _fit_tail(text: str, limit: int) -> str:
    """Cut ``text`` from the left so it fits in ``limit`` cells.

    The *tail* is kept because a partial grows to the right: the newest words
    are the ones worth seeing.  A cut is marked with a leading one-cell
    ellipsis.

    Args:
        text: Text to fit.
        limit: Available cells; ``<= 0`` yields an empty string.

    Returns:
        ``text`` unchanged when it already fits, otherwise ``"…"`` followed by
        as much of its tail as the remaining cells hold.
    """
    if limit <= 0:
        return ""
    if display_width(text) <= limit:
        return text

    budget = limit - 1  # one cell reserved for the ellipsis
    kept: List[str] = []
    used = 0
    for char in reversed(text):
        width = _char_width(char)
        if used + width > budget:
            break
        kept.append(char)
        used += width
    return _ELLIPSIS + "".join(reversed(kept))


class TranscriptPrinter:
    """Render results to a terminal without scrolling on every update.

    Partial results are transient: each one overwrites the previous on the
    same line via a carriage return, padded so a shorter update cannot leave
    the tail of a longer one behind.  A final result overwrites the same line
    and then commits it with a newline, so it scrolls away as history while the
    next utterance reuses the freed line.

    Padding and truncation are measured in *terminal cells*
    (:func:`display_width`), not characters: the transcripts are usually CJK,
    where one character draws two cells.  A line longer than the terminal is
    cut from the left rather than wrapped - a wrapped line spans two rows and
    a single ``\\r`` can no longer erase it, which would turn every update into
    a new row of leftovers.

    Args:
        stream: Where to write; defaults to ``sys.stdout``.
        width: Terminal width in cells.  Detected per write when omitted, so
            the output follows a window resize; pass a value to pin it (tests,
            or piping to a file).
    """

    def __init__(self, stream: Optional[Any] = None, width: Optional[int] = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._forced_width = width
        #: Cells drawn by the line currently on screen.
        self._drawn = 0

    def partial(self, text: str) -> None:
        """Redraw the in-progress line."""
        self._write("  ", text, end="")

    def final(self, text: str) -> None:
        """Commit a settled utterance and free the line."""
        self._write("> ", text, end="\n")
        self._drawn = 0

    def close(self) -> None:
        """End a dangling partial line, if one is on screen."""
        if self._drawn:
            self._stream.write("\n")
            self._stream.flush()
            self._drawn = 0

    def _terminal_width(self) -> int:
        """Usable cells per row, leaving the last column unwritten.

        The final column is skipped on purpose: writing into it makes some
        terminals wrap to the next row, which breaks in-place redrawing.
        """
        if self._forced_width is not None:
            columns = self._forced_width
        else:
            columns = shutil.get_terminal_size(_FALLBACK_TERMINAL_SIZE).columns
        return max(columns - 1, 1)

    def _write(self, marker: str, text: str, end: str) -> None:
        """Redraw the current line as ``marker + text``, clearing the old one.

        Args:
            marker: Two-cell prefix identifying the kind of line.
            text: Transcript text; truncated from the left when it does not
                fit next to the marker.
            end: ``""`` to keep the line for the next update, ``"\\n"`` to
                commit it.
        """
        limit = self._terminal_width()
        line = marker + _fit_tail(text, limit - display_width(marker))
        width = display_width(line)
        if width > limit:  # a terminal too narrow even for the marker
            line, width = "", 0
        padding = " " * max(min(self._drawn, limit) - width, 0)
        self._stream.write(f"\r{line}{padding}{end}")
        self._stream.flush()
        self._drawn = width


def load_wav(path: Union[str, Path], target_rate: int) -> np.ndarray:
    """Read an audio file as mono ``float32`` at ``target_rate``.

    ``soundfile`` is used rather than ``torchaudio.load``: torchaudio 2.11
    delegates decoding to ``torchcodec``, which is not a dependency of this
    repo and fails at import time.

    Args:
        path: Any file libsndfile can read (wav, flac, ...).
        target_rate: Desired sample rate in Hz.  Rate conversion goes through
            ``scipy.signal.resample_poly`` (polyphase, so it band-limits
            properly) at the ratio reduced by the greatest common divisor.

    Returns:
        A 1-D ``float32`` array.  Multi-channel input is averaged down to mono.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    import soundfile as sf

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such audio file: {path}")

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    if rate != target_rate:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), int(target_rate))
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)

    return np.ascontiguousarray(mono, dtype=np.float32)


def _import_sounddevice() -> Any:
    """Import ``sounddevice``, or exit with an actionable message.

    Raises:
        SystemExit: If the package (or the PortAudio library behind it) is
            missing.  The traceback is swallowed on purpose - a missing
            optional dependency is a setup problem, not a bug to report.
    """
    try:
        import sounddevice  # noqa: PLC0415 - optional dependency, imported late
    except Exception as exc:  # OSError too: PortAudio may be absent
        print(f"error: {_SOUNDDEVICE_HINT}\n  ({exc})", file=sys.stderr)
        raise SystemExit(1) from None
    return sounddevice


def _parse_input_device(value: Optional[str]) -> Optional[Union[int, str]]:
    """Interpret ``--input-device`` as a PortAudio index or part of a name."""
    if value is None:
        return None
    text = value.strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _wav_blocks(
    audio: np.ndarray,
    block_size: int,
    sample_rate: int,
    realtime: bool,
) -> Iterator[np.ndarray]:
    """Yield an array in blocks, optionally paced at wall-clock speed.

    Pacing is scheduled against a fixed start time rather than by sleeping a
    fixed amount per block, so time spent on inference is absorbed instead of
    accumulating into drift; a pipeline slower than real time simply stops
    sleeping.
    """
    block_seconds = block_size / sample_rate
    started = time.monotonic()
    for index, start in enumerate(range(0, len(audio), block_size)):
        if realtime:
            due = started + index * block_seconds
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        yield audio[start : start + block_size]


def _run_wav(
    session: StreamingSession,
    printer: TranscriptPrinter,
    path: str,
    realtime: bool,
) -> None:
    """Stream a file through the session as if it were live audio."""
    rate = session.config.sample_rate
    audio = load_wav(path, rate)
    print(f"streaming {path} ({len(audio) / rate:.1f}s @ {rate} Hz)")

    block_size = session.config.chunk_samples
    for block in _wav_blocks(audio, block_size, rate, realtime):
        _emit(printer, session.push_audio(block))
    _emit(printer, session.push_audio(_EMPTY_AUDIO, is_final=True))


def _run_microphone(
    session: StreamingSession,
    printer: TranscriptPrinter,
    input_device: Optional[Union[int, str]],
) -> None:
    """Capture from the microphone until interrupted.

    Args:
        session: Pipeline the captured blocks are pushed into.
        printer: Renderer for the results.
        input_device: PortAudio device index or name; ``None`` uses the
            system default input.
    """
    sounddevice = _import_sounddevice()

    rate = session.config.sample_rate
    block_size = session.config.chunk_samples
    stream = sounddevice.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        device=input_device,
        blocksize=block_size,
    )
    print(f"listening on {stream.device} - press Ctrl+C to stop")
    with stream:
        while True:
            block, overflowed = stream.read(block_size)
            if overflowed:
                # Capture outran the pipeline; the dropped audio is already
                # gone, so only say so.
                print("\n(warning: input overflow, audio was dropped)")
            _emit(printer, session.push_audio(block[:, 0]))


def _emit(printer: TranscriptPrinter, results: List["StreamingResult"]) -> None:
    """Print one batch of results."""
    for result in results:
        if result.type == "final":
            printer.final(result.text)
        else:
            printer.partial(result.text)


#: Zero-length block used to flush the pipeline at end of input.
_EMPTY_AUDIO = np.zeros(0, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Streaming SenseVoice recognition from a microphone, or from a "
            "wav file replayed in real time."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = StreamingConfig()
    parser.add_argument(
        "--model-dir",
        default="iic/SenseVoiceSmall",
        help="model id or local model directory",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help=(
            "audio capture device: a PortAudio index or part of its name; "
            "this is not a torch device (default: the system default input)"
        ),
    )
    parser.add_argument(
        "--torch-device",
        default=defaults.device,
        help=(
            "torch device the models run on (cpu, cuda:0, ...); this is not "
            "an audio device"
        ),
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
        "--wav",
        default=None,
        help="read this audio file instead of the microphone",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="with --wav, run as fast as possible instead of at 1x speed",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print the available audio input devices and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run the demo.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit status: ``0`` on success or ``Ctrl+C``, ``1`` when the
        input cannot be read, ``2`` when the configuration is rejected.
    """
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print(_import_sounddevice().query_devices())
        return 0

    config = StreamingConfig(
        chunk_size=args.chunk_size,
        max_history=args.max_history,
        device=args.torch_device,
    )
    try:
        config.validate()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from streaming.streaming_model import StreamingSenseVoice

    print(f"loading {args.model_dir} ...")
    session = StreamingSession(
        model=StreamingSenseVoice(args.model_dir, config),
        vad=VadGate(config),
        config=config,
    )

    printer = TranscriptPrinter()
    try:
        if args.wav:
            _run_wav(session, printer, args.wav, args.realtime)
        else:
            _run_microphone(session, printer, _parse_input_device(args.input_device))
    except KeyboardInterrupt:
        _emit(printer, session.push_audio(_EMPTY_AUDIO, is_final=True))
        printer.close()
        print("stopped")
        return 0
    except FileNotFoundError as exc:
        printer.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    printer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
