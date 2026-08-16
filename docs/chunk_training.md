# Chunk training and streaming inference

Status: **experimental.** The code paths described here are implemented and
tested, but no chunk-finetuned checkpoint exists yet and no streaming WER or
latency measurement of the chunk backend has been taken. Everything below
distinguishes what has been *measured* from what is *reasoned*.

## Why this exists

The published `SenseVoiceSmall` weights are trained with full self-attention:
every frame attends to the whole utterance. The existing streaming recogniser
(`backend="accumulate"`) works around this by re-running the full encoder over a
growing window on every partial. That is correct but costly — the encoder has a
large fixed cost per pass (~430 ms on CPU), so cost grows with utterance length
and the same frames are re-encoded many times.

Chunk training restricts self-attention to a sliding window during finetuning, so
the resulting model can be decoded incrementally: each frame is encoded exactly
once, against a cache. That is what `backend="chunk"` does.

**The catch:** running the *published* weights through the chunk path is a
different computation from what they were trained for, and quality will degrade.
The chunk backend is only expected to be useful after a chunk finetune. See
"What is verified" below for how large the gap is.

## The two code paths

Both live on `SenseVoiceEncoderSmall` in `model.py`.

| | `forward` (training) | `forward_chunk` (inference) |
|---|---|---|
| Enabled by | `chunk_size=...` at construction | `init_chunk_cache()` + a cache |
| Attention | restricted by a chunk mask | restricted by the window + KV cache |
| Sequence length | expanded, then folded back | one window per call |
| Default | `chunk_size=None` → unmodified full attention | n/a |

With `chunk_size=None` the encoder is byte-for-byte the original full-attention
model, so nothing here affects existing use.

### Chunk masking during training

`forward` builds a funasr `overlap_chunk` and splits the sequence into
overlapping chunks. Passing several `chunk_size` entries enables **dynamic chunk
training**: one entry is drawn per step, so a single checkpoint learns several
latency operating points. An entry `<= 0` is a **full-attention sentinel** — when
drawn, that step runs the plain full-attention path, which is how full-attention
batches are mixed in to stop the model over-fitting to one geometry.

Unlike funasr's `SANMEncoderChunkOpt`, the chunk-expanded sequence is folded back
onto the original time axis (via `remove_chunk`) before returning. This is
deliberate: it means `olens` equals the input lengths, so the CTC head and the
loss need no modification at all. That property is pinned by a test.

### Streaming inference

`forward_chunk` encodes one chunk against a cache carrying the position-encoding
offset, a frame overlap, and optional per-layer attention key/value caches.

The frame overlap exists because **the SANM memory block has no streaming state
of its own** — its context comes purely from chunk overlap. Frames within
`kernel_size // 2 == 5` of a window edge therefore see zero padding where the
full-attention path sees real neighbours.

## Chunk geometry — read this before configuring anything

There are **two different 3-tuple conventions** in play, and confusing them is
the easiest mistake to make here.

| Where | Meaning |
|---|---|
| `encoder_conf.chunk_size` (training) | **total window width** = `pad_left + stride + pad_right` |
| `cache["chunk_size"]` (inference) | **`[pad_left, stride, pad_right]`** |

The inference convention matches funasr's `init_cache` default `[0, 10, 5]`. Use
`init_chunk_cache()` to derive the cache tuple from the training configuration
rather than writing it by hand.

One encoder frame is **60 ms** (10 ms mel hop x LFR factor 6). So for the
configuration shipped in `finetune_chunk.sh`:

| entry | `chunk_size` | `stride` | `pad_left` | implied `pad_right` | window | commit | lookahead |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 6 | 0 | 2 | 480 ms | 360 ms | 120 ms |
| 1 | 12 | 10 | 0 | 2 | 720 ms | 600 ms | 120 ms |
| 2 | 16 | 14 | 0 | 2 | 960 ms | 840 ms | 120 ms |

`pad_right` is what costs latency: a frame is only emitted once its `pad_right`
successors have been fed.

Constraints enforced at construction: `0 < stride <= chunk_size` and
`0 <= pad_left <= chunk_size - stride`. Additionally,
`encoder_chunk_look_back != 0` requires `pad_right >= 1`, because the attention
cache is built by dropping the last `pad_right` frames and would otherwise be
empty. Combining look-back with `pad_left > 0` also re-appends left-context
frames to that cache (upstream funasr behaviour), so prefer `pad_left = 0`
whenever look-back is enabled.

## Running a real finetune

> None of the numbers in this section are measured on this repository. They are
> starting points derived from the model size and the existing `finetune.sh`
> defaults. Treat them as a hypothesis to validate, not a recipe.

### Hardware

SenseVoiceSmall is ~234M parameters. `finetune_chunk.sh` defaults to
`CUDA_VISIBLE_DEVICES=0,1` with `batch_type=token, batch_size=6000`. Chunk
masking **expands the sequence** before folding it back, so peak activation
memory is higher than a full-attention step at the same batch size — expect to
reduce `batch_size` relative to `finetune.sh` if you hit OOM, rather than
assuming parity.

Deepspeed config is wired up (`deepspeed_conf/ds_stage1.json`) but
`use_deepspeed=false` by default.

### Data

Chunk finetuning is an adaptation of an already-trained model, not training from
scratch. The model must learn to work without right context, which is a
comparatively small behavioural change. Order-of-magnitude expectation: tens to
hundreds of hours of in-domain speech, far less than the original training set.
Use `data/train_example.jsonl` for the manifest schema.

Keep a full-attention sentinel (`-1`) in `chunk_size` if you want the finetuned
checkpoint to remain usable for offline full-attention decoding.

### Epochs and learning rate

`finetune_chunk.sh` inherits `max_epoch=50` and `lr=0.0002` from `finetune.sh`.
For an adaptation run, fewer epochs at a lower LR is the safer starting point —
the risk is catastrophic forgetting of the full-attention behaviour, not
underfitting. Watch validation loss on a held-out set decoded *in chunk mode*;
full-attention validation loss will not show the degradation you care about.

### Choosing the geometry

Decide the latency budget first, then pick `pad_right`. Lookahead is the only
parameter that directly buys accuracy at the cost of latency. Train with several
`chunk_size` entries so one checkpoint serves several operating points, then pick
the decode geometry from the entries actually trained on — that is what the
streaming config defaults do.

### CPU smoke check

`SMOKE=1 ./finetune_chunk.sh` runs a few-step CPU pipeline check on locally
generated dummy data. It validates plumbing only — the transcripts are dummies,
so **nothing about the resulting model is meaningful**.

Measured: 6 optimizer steps, exit 0, ~54 s wall clock, losses finite
(ctc 66.4–107.6, rich 0.26–0.89), chunk config confirmed active in the trainer's
dumped kwargs.

Known limitation of the smoke path: it runs funasr's `train.py`, **not** the
`train_ds.py` used by the GPU path. `train_ds.py` unconditionally reassigns
`kwargs["device"]` and `trainer.device` to `int(LOCAL_RANK)` right after
`warp_model`, sending every batch to accelerator index 0 regardless of config, so
it cannot run on CPU at all. The smoke therefore validates the model, dataset and
chunk path — not `train_ds.py`'s own logic.

## What is verified, and what is not

Pinned by `tests/test_chunk_streaming_equivalence.py` (small randomly-initialised
encoder, fixed seed, no checkpoint):

**Exact:**
- A single `forward_chunk` call covering the whole utterance is **bit-identical**
  to `forward` (max diff exactly 0.0, both float32 and float64).
- The chunk training path restores the original time axis, so `olens` equals the
  input lengths.
- A `chunk_size=(-1,)` sentinel encoder matches a `chunk_size=None` encoder
  exactly.
- Already-emitted frames never change when the utterance is extended (max diff
  0.0) — output handed to the caller is final.
- The per-call emission schedule is a pure function of geometry and length:
  first call emits `stride - pad_right`, later calls emit their input length, the
  tail flush emits `pad_right`, and the total equals the input length.

**Deliberately approximate:** a multi-chunk stream does **not** equal full
attention, and is not expected to. Six divergence sources are enumerated in that
file's module docstring; four are structural, one is inert, one is bounded well
below the others. The deviation is asserted from *both* sides — an upper bound
catches regressions, a lower bound catches streaming silently degenerating into
full attention. float32 and float64 agree on the deviation to ~4e-7 relative,
which is what licenses reading it as algorithmic rather than rounding noise.
Unbounded look-back measurably narrows the gap but can never close it, because
the right-hand context stays capped at `pad_right`.

**Important caveat on magnitudes:** those deviation figures come from a small
randomly-initialised encoder, not the pretrained checkpoint. They establish the
*structure* of the divergence, not how much word error it causes. The quality
cost of chunked decoding on real weights is **unmeasured**.

### Real-weight smoke (published, non-finetuned checkpoint)

One run on `runtime/llama.cpp/tests/sample.wav` (~6 s, Chinese), published
`SenseVoiceSmall` weights, both backends fed identical audio:

- **Runs end to end, no crash**, on either backend.
- **Finals are identical** across `accumulate`, `chunk`, and the offline
  `model.inference()` reference: `我想问我在滨海新区有房。` This is expected —
  `_infer_final` is backend-independent — and confirms the shared full-quality
  path is intact.
- **Chunk partials are degraded but recognisable**, not garbage. Final chunk
  partial: `我想想问我在滨海心新区有` versus the reference
  `我想问我在滨海新区有房`. The errors are duplicated (`我想想问`) and spurious
  (`滨海心新区`) characters — the expected cost of truncated attention against a
  checkpoint never finetuned for it.
- **The chunk backend was *slower* on this clip**, not faster: RTF 0.818 vs
  0.697, max `push_audio` 879 ms vs 493 ms. At 6 s the accumulate window never
  reaches `max_history=167`, so the chunk backend's constant-per-frame advantage
  is not exercised at all. **"Chunk is cheaper" is unmeasured** — it should only
  be expected to pay off on utterances long enough for the accumulate window to
  saturate.

This is a single clip in one language and is not a quality measurement.

**Not verified at all:**
- Streaming WER, of either backend, against any reference.
- The crossover length at which the chunk backend actually becomes cheaper.
- Chunk-backend latency and CPU occupancy at scale (the `accumulate` defaults are
  benchmarked; the chunk geometry defaults are not).
- That a chunk finetune actually recovers quality — no such checkpoint exists.

## Deviation from upstream funasr

`init_chunk_cache` seeds `cache["feats"]` with `pad_left` zero frames. Upstream
funasr (`funasr/models/scama/model.py`, `init_cache`) seeds
`pad_left + pad_right`.

This is a **fix, not a tolerated regression.** On the very first call nothing has
been withheld yet, so upstream's extra `pad_right` frames stand for a lookahead
that does not exist. They are not inert — they enter self-attention and the FSMN
convolution. With upstream's seed the first call emits `stride` frames instead of
`stride - pad_right`, so the stream carries `pad_right` phantom output frames for
the whole utterance and no longer aligns one-to-one with its input, which is the
contract CTC decoding depends on.

Measured (group B of `tests/test_chunk_streaming_equivalence.py`):

- The two first-call layer-0 windows **cannot** be compared by a bare max
  difference — upstream's is exactly `pad_right` frames longer.
- Those surplus leading frames are exactly `0.0`.
- After realigning (dropping them) the windows are **bit-identical**, max
  difference `0.0`, across all six geometries tested:
  `(0,10,0) (0,10,5) (5,10,0) (5,10,5) (3,8,4) (0,6,2)`.
- The resulting output corruption is confined to the first emitted chunk, peaking
  at `1.514661` for `(0,10,5)`, `0.370553` for `(5,10,5)`, `1.244551` for
  `(0,8,4)`. Every frame from index `stride` onward is bit-identical.

Everything else follows upstream. In particular `sanm_shfit=0` is kept so
published checkpoints still load, and the added
`StreamSinusoidalPositionEncoder` holds no parameters or buffers, so
`state_dict()` is unchanged and checkpoints load with `strict=True`.

Because this depends on funasr internals that are private API of a pinned
release, `tests/test_chunk_training_mask.py` pins the `overlap_chunk` behaviour
this relies on. If you bump funasr, that file is what will tell you what broke.

## Streaming backends

`streaming/config.py` selects the strategy with `backend`:

| | `"accumulate"` (default) | `"chunk"` |
|---|---|---|
| Encoder work | re-runs the full encoder over a growing window | each frame encoded once into a cache |
| Cost | grows with utterance length | constant per frame, but higher fixed cost |
| Weights | works with published weights | expects a chunk-finetuned checkpoint |
| Measured | yes (CPU benchmark) | no |

The cost row is asymptotic, not a promise: on a 6 s clip the chunk backend
measured *slower* than accumulate (see the real-weight smoke above). It should
only be expected to win once utterances are long enough for the accumulate
window to saturate `max_history`.

The default is unchanged and remains the only path with measured numbers.

Chunk geometry is configured with `chunk_pad_left`, `chunk_stride`,
`chunk_pad_right` and `chunk_encoder_look_back`; the defaults mirror the middle
entry of the `finetune_chunk.sh` training configuration, on the reasoning that
decoding should use a geometry the encoder was actually trained on. **They are
not tuned** — treat them as a starting point to benchmark.

**Reachability:** `streaming/ws_server.py` has no `--backend` flag, so the chunk
backend is currently only selectable in code, not from the shipped WebSocket
entrypoint. That is deliberate while it is unbenchmarked — add the flag when
there is a finetuned checkpoint and a measurement to justify it.

Note that `chunk_size` in `StreamingConfig` is *not* the encoder's chunk width.
It is the emission cadence — how many new frames must arrive before a partial is
produced — and applies to both backends. The chunk backend's encoder window is
`chunk_pad_left + chunk_stride + chunk_pad_right`, stepped several times per
emitted partial when the two differ.

### Rich labels in the chunk backend

SenseVoice emits its language / emotion / event tags at the first few output
positions. A chunk stream only has the first chunk's audio when those positions
are produced, so the tags in a **partial are provisional**. The `final` result
comes from the shared full-quality pass over the retained waveform and is
authoritative. Do not treat a partial's rich tags as settled.
