# Chunk training and streaming inference

Status: **experimental, with one finetuned checkpoint.** The code paths
described here are implemented and tested, and a Japanese chunk finetune has
now been run end to end — it closes 90% of the chunk-versus-full CER gap
(0.2433 → 0.0245) without degrading full attention. Streaming WER and
chunk-backend latency at scale remain unmeasured. Everything below
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

## The Japanese chunk finetune (Apple Silicon / MPS)

> Status: **run completed.** A chunk-finetuned checkpoint now exists and the
> quality question this document previously listed as unmeasured is answered
> for Japanese. See "Results" at the end of this section.

**Scope: this finetune is Japanese-specialised. Multilingual performance is
explicitly _not_ a maintained property.** The base checkpoint covers
zh/ja/yue/en/ko, but this adaptation trains on Japanese only, so degradation in
the other four languages is accepted and untested. Checkpoint selection is
decided on Japanese quality alone. If multilingual retention is ever wanted,
that is a different run with a mixed-language corpus, not this one.

### Corpus

[VisualNovel_Dataset](https://huggingface.co/datasets/OOPPEENN/56697375616C4E6F76656C5F44617461736574)
(MIT), 597 password-protected `.7z` archives, 478.5 GB, 10,588 h of Japanese
visual-novel speech. Each archive holds `index.json` — a list of
`{Speaker, Voice, Text}` — plus `<Speaker>/<Voice>.ogg` at 48 kHz, mixed mono
and stereo, mean 4.0 s. There is no streaming access: the minimum download unit
is one whole archive.

Five archives were selected for ~53.8 h across ~200 speakers, chosen by
hours-per-byte so the download stays at 1.28 GB. `scripts/prepare_vn_data.py`
downloads, extracts, resamples to 16 kHz mono and emits the manifest schema of
`data/train_example.jsonl`. The split is **speaker-disjoint** — held-out
speakers appear in no training utterance — because speaker leakage would make
the held-out chunk-mode decode meaningless.

The corpus is unfiltered adult content and contains NSFW transcripts. These are
deliberately *not* censored: an ASR target must match its audio, and rewriting
transcripts would simply teach the model to mistranscribe.

### Why not moe-speech

[litagin/moe-speech](https://huggingface.co/datasets/litagin/moe-speech) (623 h
of studio-recorded Japanese character speech) was evaluated and **rejected: it
contains no transcripts at all.** Its own README says so — "this dataset doesn't
contain any text information" — and its task tags list TTS, voice conversion and
speaker identification, with no ASR task. It is unusable for the ground-truth
supervised finetune chosen here.

It would become usable under a self-distillation objective, where the
full-attention model's own output supplies the target and no ground truth is
needed. That approach was considered and not taken. Note also its licence:
Japanese Copyright Act Art. 30-4 (machine-learning use only), redistribution of
even a single audio file prohibited, though publishing a trained model is
explicitly permitted.

### Running it on MPS

`finetune_chunk_mps.sh`. The GPU path (`train_ds.py`) cannot be used — see the
smoke-check limitation above, it forces every batch to accelerator index 0 — so
the MPS launcher extends the `train.py` path instead.

**`PYTORCH_ENABLE_MPS_FALLBACK=1` is mandatory.** Without it training dies with
`NotImplementedError: aten::_ctc_loss is not currently implemented for the MPS
device`. That is the *only* blocking gap: the SANM/FSMN encoder and the chunk
masking both run as native MPS kernels. CTC loss and its backward execute on
CPU via the fallback, agreeing with a CPU reference to 1.25e-7 relative and
costing about 2% of step time, though they do force a host sync every step.

Measured on an M5 Pro (64 GB unified memory, torch 2.13.0, fp32, chunk masking
on, 2–10 s utterances):

| device | batch | s/step | audio-h per wall-h | MPS driver alloc |
|---|---|---|---|---|
| cpu | 4 | 4.30 | 5.6 | — |
| mps | 4 | 0.91 | 26.5 | 15.8 GB |
| mps | 8 | 1.42 | 33.9 | 26.2 GB |
| mps | 16 | 2.70 | 35.5 | 44.7 GB |

MPS is **4.75x faster than CPU** at matched batch. The binding constraint is
memory, not compute: the recommended working-set cap is 55.7 GB, and past
batch ~16 throughput collapses rather than scales. Because chunk masking
expands the sequence, batching is capped by **audio seconds per step (~72 s)**
rather than funasr's GPU-oriented `batch_type=token` — which works out to
`batch_size=7200` with `batch_size_sample_max=12`, i.e. at most 12 clips per
step and ~35 GB in practice. Batch 8 is the conservative alternative if a
longer-utterance corpus pushes memory up: it keeps 96% of batch 16's
throughput for 59% of the memory.

`model.half()` fails on both MPS and CPU, and the cause is in this repository
rather than in MPS: `sequence_mask()` in `model.py` hardcodes
`dtype=torch.float32`, promoting half activations back to float32 and colliding
with the FSMN convolution. Use autocast if mixed precision is wanted; the run
described here is fp32 and does not touch `model.py`.

### Run configuration

53.8 h, 4 epochs, ~6 h wall clock. LR **2e-4 is a ceiling, not a target** — it
is inherited from `finetune.sh` and may be lowered if evaluation shows
degradation, but must not be raised. Chunk geometry is unchanged from
`finetune_chunk.sh` (`chunk_size=[8,12,16]`, `stride=[6,10,14]`,
`pad_left=[0,0,0]`, look-back `[1,1,1]`). Checkpoints land in `OUTPUT_DIR`
(default `outputs/chunk_mps/`; the first real run used `outputs/chunk_mps_run1/`
to keep it clearly separate from earlier pipeline-check artefacts).
`keep_nbest_models` is set high enough to disable pruning, so **every epoch
checkpoint is retained** — plus intra-epoch ones every 1000 steps — and the best
is chosen after the fact rather than trusting the last. Budget ~2.8 GB per
checkpoint (the extra over the 0.94 GB base is AdamW optimiser state), so a
4-epoch run leaves roughly 45 GB behind.

`OUTPUT_DIR` has **no lock or PID guard**: two runs pointed at the same
directory will both write `model.pt` and both prune, producing torn
checkpoints. Give every run its own directory. For the same reason, do not run
`eval_chunk_gap.py` with `--device mps` while a training job is live — the two
will contend for the same memory budget. Use `--device cpu` mid-run, or wait.

`scripts/eval_chunk_gap.py` runs per epoch and reports, on Japanese only:
chunk-mode decode quality on the held-out speakers, full-attention quality on
the same clips (the forgetting check), and the chunk-versus-full gap that the
whole exercise exists to close. Per the guidance above, validation is a
chunk-mode **decode** — full-attention validation loss does not show the
degradation that matters here.

**Ignore funasr's own checkpoint selection.** This run ended with
`Update best acc: 0.0000 -> model.pt.best` and
`average_checkpoints: ['model.pt.ep0.1000']  -> model.pt.avg1`: `best` was
chosen on an accuracy metric that is identically zero for this model, and
`avg1` averaged a single early intra-epoch checkpoint. Neither is meaningful.
Select on the chunk CER reported by `eval_chunk_gap.py`.

### Results

52.99 h corpus (29,230 train / 772 val clips, 128 vs 50 speakers,
speaker-disjoint), 4 epochs, batch 12, LR 2e-4, fp32, **4.5 h wall clock** on an
M5 Pro. Corpus-level CER on the 772 held-out Japanese clips, decoded at
geometry index 1 (`chunk_size=12, stride=10, pad_right=2`, 120 ms lookahead):

| checkpoint | chunk CER | full-attention CER | gap |
|---|---|---|---|
| base (published) | 0.4059 | 0.1625 | 0.2433 |
| epoch 1 | 0.2140 | 0.1615 | 0.0525 |
| epoch 2 | 0.1851 | 0.1493 | 0.0359 |
| epoch 3 | 0.1782 | 0.1475 | 0.0307 |
| **epoch 4 (best)** | **0.1739** | **0.1494** | **0.0245** |

**The chunk finetune works.** Chunk CER falls 57% relative (0.4059 → 0.1739)
and the chunk-versus-full gap closes 90% (0.2433 → 0.0245). Most of the gain
lands in the first epoch, which matches the "adaptation, not retraining"
framing above.

**No catastrophic forgetting on Japanese** — full-attention CER did not
degrade; it *improved* slightly (0.1625 → 0.1494), because 52 h of in-domain
visual-novel speech helps full attention on this domain too. Both curves were
still improving at epoch 4, so the run was stopped by budget rather than by
convergence; more epochs are the obvious next experiment.

**Not measured, and deliberately so:** performance in zh/yue/en/ko. Training
was Japanese-only and multilingual retention is not a maintained property of
this checkpoint (see the scope note at the top of this section). The Chinese
clip in `runtime/llama.cpp/tests/sample.wav` is carried by
`eval_chunk_gap.py` as an informational reference only and does not participate
in checkpoint selection.

Also still unmeasured: streaming WER against any reference, the crossover
length at which the chunk backend becomes cheaper than `accumulate`, and
chunk-backend latency at scale. A finetuned checkpoint now exists, so these
are finally answerable.

### Known gaps in the tooling

- `scripts/eval_chunk_gap.py` hardcodes the chunk geometry and selects it by
  index, duplicating the values in both finetune scripts. Training now writes
  the resolved geometry to `${OUTPUT_DIR}/chunk_geometry.json` and warns when
  it deviates from the default, but evaluation does not yet read that file —
  so a non-default training geometry still has to be mirrored into the
  evaluator by hand or the measurement is invalid.
- `finetune_chunk_mps.sh` takes an exclusive `OUTPUT_DIR` lock; the GPU
  `finetune_chunk.sh` does not, and so remains vulnerable to two concurrent
  runs shredding each other's checkpoints.

## The 300-hour cluster finetune (H100 / Slurm)

> Status: **runs completed.** Two full training runs finished on an H100 NVL
> cluster and both beat the MPS baseline's chunk CER. See "Results" below.

Same scope note as the MPS section: **Japanese-specialised, multilingual
retention is not a maintained property.**

`finetune_chunk_slurm.sh` is the cluster sibling of `finetune_chunk_mps.sh`. It
follows the GPU path of `finetune_chunk.sh` (torchrun onto funasr's
`bin/train_ds.py`) and keeps the same chunk geometry, `OUTPUT_DIR` lock and
preflight, so the two runs stay comparable.

### Cluster facts you cannot guess

Every item here was found by a job failing, and none of it is discoverable from
the site documentation. If you are adapting this to another Slurm site, these
are the assumptions to re-test rather than inherit.

- **The scheduler discards the batch script's exit code.** A job whose entire
  body is `exit 42` is recorded by `sacct` as `COMPLETED ExitCode 0:0`. *Every*
  job reports success, including one that crashed in its first minute. Job state
  is therefore worthless as a success signal. `finetune_chunk_slurm.sh` writes
  `${OUTPUT_DIR}/.job_status` and ends its log with `SENSEVOICE_JOB_OK` or
  `SENSEVOICE_JOB_FAILED rc=<n>`; those are the only trustworthy outcomes. This
  is also why `scripts/submit_chunk_chain.sh` chains with `afterany` — `afterok`
  would fire unconditionally and hide a failed link.
- **`/data` is unusable as a container mount point.** The site mounts a local
  scratch filesystem (`/dev/md0`) there *after* the bind mounts are applied, so
  a `-v host:/data` silently resolves to someone else's data. It fails
  invisibly: the directory exists, is readable, and lists plausible content. The
  corpus is mounted at `/corpus` for this reason.
- **`SLURM_*` variables are unset inside the container**, so GPU count is
  derived from `CUDA_VISIBLE_DEVICES` (which *is* set) and never from
  `SLURM_GPUS_PER_TASK`. Ordinary environment variables *do* reach the
  container, so `SMOKE=1 sbatch ...` works; script *arguments* do not.
- **Slurm executes a spool copy of the batch script**, so `BASH_SOURCE[0]`
  points at `/tmp/slurm/...`, not the repo. Export `WORKSPACE=/workspace`.
- **The manifest stores absolute container paths.** Changing the mount point
  means regenerating the manifests, not just remounting.
- The site `sbatch` is a wrapper that reads `#SBATCH --container=` and
  `#CONTAINER` lines and docker-pulls the image before submitting. The verified
  GRES syntax is `--gpus-per-task=N` with `--nodes=1 --ntasks=1`.

### Image

`docker/Dockerfile.cluster`, built on the site's NGC PyTorch mirror.

**The NGC base ships no torchaudio at all**, and funasr needs it
(`funasr/frontends/wav_frontend.py` imports `torchaudio.compliance.kaldi`).
Installing torchaudio alone against NGC's torch fails at load time —
`libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv` — because
NGC builds torch with a custom ABI that no PyPI torchaudio wheel links against.
The image therefore installs a **matched PyPI pair**, `torch==2.13.0` +
`torchaudio==2.11.0` (cu126), replacing NGC's torch. This is not a compromise:
those are the versions in the project's own `.venv` that produced the MPS
baseline, and they satisfy `requirements.txt`, whereas NGC's torch 2.6 is
*older* than what this project targets. NGC's `torchvision` and `torch_tensorrt`
are uninstalled because they hard-pin the replaced torch.

Two base-image quirks are worked around: `pip check` crashes on malformed
`.egg` wheel tags (reproduced in the pristine image), so a targeted
`importlib.metadata` requirement check replaces it; and a dead
`extra-index-url` costs ~16 s of DNS retries per package, so it is stripped from
**all five** pip config paths — the base ships the same dead index twice, and
the user-level copy outranks the global one.

The build verifies `torch.version.cuda`, `kaldi.fbank`, and
`import funasr.bin.train_ds`. That last check is what caught the missing
torchaudio, before any job was queued.

### Corpus

300.0 h drawn from the same VisualNovel_Dataset, 24 archives (11.9 GB) across
**22 studios**. Studio diversity was the selection criterion rather than
hours-per-byte: filling the budget from the top of an alphabetical listing
collapses onto two or three studios and narrows speaker, direction and recording
variety. The 5 archives of the 53.8 h MPS corpus are included, so this corpus is
a strict superset of the baseline's.

| | clips | hours |
|---|---|---|
| train | 196,047 | 282.93 |
| val | 5,194 | 7.48 |

The split is speaker-disjoint, verified independently of the script by reading
the emitted manifests: **0 speaker overlap** (450 train / 409 val speakers) and
**0 audio-path overlap**. 4.7 % of val transcripts also occur in train — these
are frequent short utterances ("はい", "ええ"), not leakage, since audio and
speakers are disjoint; they make the CER slightly easier and are noted here
rather than removed.

`--limit-hours 300` cut the corpus at exactly 300.000 h, skipping 12,997
surplus clips. `dropped_missing_audio: 14,112` is an upstream dataset property,
not a truncated extraction: every one of the 24 archives was verified with 7z
entry count == on-disk file count (delta 0).

Fingerprints of the manifests these runs used:

```
train.jsonl     ba360aee120794e301e86477c98ba7b59c6cc8ce02c13508f2a0b7cde021d3d6
val.jsonl       a4e3167e56d09f954b46addb58a793d4dc9502f45e297a9ba29ef246038c190e
manifest.json   30adf4d284f6982c9b2d7b9072b3ed1534e48966503fefd21bf99fed6d9d52c0
```

### Run configuration

Batch size has **two** limits and the smaller one binds. With `BATCH_TOKENS`
6000 and `MAX_SAMPLES_PER_STEP` 12, a mean clip of 5.20 s (520 units) means
12 clips ≈ 6,240 units — the two are already balanced, so raising
`BATCH_TOKENS` alone changes nothing at all. Both must move together.

Measured on one H100 NVL (94 GB), on the real corpus:

| tokens / samples | step time | peak VRAM | throughput |
|---|---|---|---|
| 24,000 / 48 | 0.19 s | 32.3 GB (34 %) | 1,263× realtime |
| 48,000 / 96 | 0.30 s | 72.9 GB (78 %) | 1,600× realtime |

Doubling the batch bought 27 % throughput — already well into diminishing
returns — while leaving little headroom for an outlier batch of long clips, so
**24,000 / 48** was chosen. One epoch is ~15 min on a single GPU; a 4-epoch run
on 2 GPUs takes **~45 min**. The 24 h wall clock and the resume-chain machinery
built for this run were never needed.

`warmup_steps` deserves attention. It is inherited from the base model's
`config.yaml` as 25,000, and `warmuplr` is Noam-style — it peaks at
`warmup` then decays as `step^-0.5`. The MPS run's 12,720 total steps never
reached the peak, so it effectively trained on a linear ramp topping out near
1.0e-4 rather than at its nominal 2e-4. These runs set
`WARMUP_STEPS=680` (~8 % of 8,488 steps) so the schedule actually peaks and
decays.

**The learning rate was chosen by measurement, not by rule.** Because a run
costs under an hour, two conditions were trained in parallel within the QOS
budget: LR 2e-4 (baseline-equivalent) and LR 6e-4 (square-root scaling of the
8× larger effective batch). Square root, not linear scaling, because the failure
mode to fear when adapting a pretrained checkpoint is catastrophic forgetting;
linear scaling would have suggested 3.2e-3.

### Results

Corpus-level CER on held-out Japanese, geometry index 1
(`chunk_size=12, stride=10, pad_right=2`, 120 ms lookahead), decoded in fp32
with TF32 explicitly disabled (`allow_tf32=false`, `cudnn.deterministic=true`)
so the numbers are comparable with the fp32 baseline. Selection sweep, 800 clips
of the val set, identical subset for every checkpoint:

| run | checkpoint | chunk CER | full CER | gap |
|---|---|---|---|---|
| base (published) | — | 0.3880 | 0.1663 | 0.2217 |
| LR 6e-4 | **epoch 4 (best)** | **0.1624** | **0.1489** | 0.0135 |
| LR 6e-4 | epoch 3 | 0.1648 | 0.1569 | 0.0079 |
| LR 2e-4 | epoch 4 (best) | 0.1655 | 0.1521 | 0.0134 |
| LR 2e-4 | epoch 2 | 0.1656 | 0.1533 | 0.0123 |
| LR 2e-4 | epoch 3 | 0.1680 | 0.1570 | 0.0110 |
| LR 6e-4 | epoch 2 | 0.1717 | 0.1613 | 0.0104 |

All eight checkpoints beat the MPS baseline's 0.1739 chunk CER on this subset.

The top candidates were then re-scored on the **complete 5,194-clip val set**.
These are the numbers to quote:

| model | chunk CER | full CER | gap |
|---|---|---|---|
| base (published) | 0.4024 | 0.1781 | 0.2243 |
| LR 2e-4, epoch 4 | **0.1679** | 0.1584 | **0.0095** |
| LR 6e-4, epoch 4 | 0.1687 | **0.1550** | 0.0138 |
| LR 6e-4, epoch 3 | 0.1718 | 0.1616 | 0.0102 |

Against the base model on the same val set — the only strictly like-for-like
comparison available — **chunk CER falls 58 % relative** (0.4024 → 0.1679) and
**the chunk-versus-full gap closes 96 %** (0.2243 → 0.0095).

**The two learning rates are indistinguishable, and the 800-clip sweep said
otherwise.** On the subset, 6e-4 led 2e-4 by 1.9 % relative; on the full set the
order reverses and the margin is 0.5 % (0.1679 vs 0.1687). A 0.5 % gap is not
something 800 clips can resolve, so the subset ranking between these two was
noise — it was adequate for separating good checkpoints from bad (the spread
there was 0.1624–0.1717) but not for splitting the top two. **Treat the
selection sweep as a filter, not as a verdict**, and re-score finalists on the
whole set before claiming a winner. The square-root LR scaling neither helped
nor hurt at this batch size.

The subset misled on the gap as well: it made LR 6e-4 epoch 3 look like the
smallest-gap checkpoint by a clear margin (0.0079), and on the full set that
checkpoint is worse than the adopted one on every measure (0.1718 / 0.1616 /
0.0102). Two independent claims from the same 800 clips both failed to
replicate, which is the strongest argument here for scoring finalists on the
whole set.

The two epoch-4 checkpoints differ in *where* they are better, and that much is
a real choice rather than noise: 2e-4 has the smaller chunk-vs-full gap
(0.0095), 6e-4 retains more full-attention quality (0.1550 vs 0.1584).
Selection on chunk CER alone picks 2e-4 by a hair; picking 6e-4 to minimise
forgetting is equally defensible.

**Adopted: LR 2e-4, epoch 4.** Two reasons. The pre-registered selection metric
is chunk CER (`eval_chunk_gap.py` records it as `ja_val_chunk_cer` in every
report), and switching criteria *after* seeing the results — immediately after
a subset ranking had already misled us once — is how a tie gets talked into
whichever answer is convenient. And the product metric here is streaming
quality, which the smaller chunk-vs-full gap serves directly. The full-CER
advantage of 6e-4 is real but small enough to accept.

**No catastrophic forgetting.** Full-attention CER at the adopted checkpoint is
0.1584 against the base model's 0.1781 on the same val set. Streaming quality
did not come out of offline quality.

Extract the deployable weights with:

```
python scripts/extract_weights.py <output_dir>/model.pt.ep4 weights_ep4.pt
```

which drops optimizer state (2,679 MiB → 893 MiB) and writes a flat
`OrderedDict` matching the published `model.pt` key-for-key, so it is a drop-in
replacement.

**No catastrophic forgetting.** Full-attention CER at the winning checkpoint is
0.1489, better than the base model's 0.1663 and marginally better than the MPS
baseline's 0.1494. The chunk-mode gain did not come out of offline quality.

Note that epoch 3 of the LR 6e-4 run has the *smallest* gap (0.0079) but a worse
full CER. Selection is on chunk CER, which is the quantity that matters for
streaming; the gap is diagnostic, not the objective.

### How this compares to the MPS baseline

The two numbers are **not measured on the same val set** — each is the held-out
val of its own corpus, and the 300 h val is larger and drawn from 22 studios
rather than 5. Treat the comparison as directional. The differences that matter:

| | MPS baseline | cluster run |
|---|---|---|
| corpus | 53.8 h, 5 archives | 300.0 h, 24 archives, 22 studios |
| val | 772 clips | 5,194 clips |
| device / precision | MPS, fp32 | H100 NVL, bf16 |
| effective batch | 6,000 units | 48,000 units (2 GPUs × 24,000) |
| total steps | 12,720 | 8,488 |
| warmup | 25,000 (never reached; effective peak ~1.0e-4) | 680 (peaks, then decays) |
| LR | 2e-4 | 2e-4 and 6e-4 compared |
| wall clock | 4.5 h | ~45 min per condition |

An earlier plan to re-score the winning checkpoint on the *old* 772-clip val was
abandoned after measurement: because the new corpus is a superset, **547 of
those 772 clips (70.9 %) are in the new training set**. Scoring there would have
been scoring on training data.

### Known gaps from this run

- **`avg_nbest_model` and `keep_nbest_models` conflict.** Pruning runs before
  averaging, so the averager looks for a checkpoint that pruning already
  deleted, logs `No checkpoints found for averaging`, and produces no
  `model.pt.avg1`. Reproduced identically in the smoke run and both full runs.
  Weights are recovered with `scripts/extract_weights.py` instead
  (2,679 MiB → 893 MiB, optimizer state dropped).
- `eval_chunk_gap.py` re-scores the base model once per checkpoint, roughly
  doubling sweep time. A cached base result or a multi-checkpoint mode would
  halve it.
- The `#CONTAINER` mount lines and the image reference in
  `finetune_chunk_slurm.sh` cannot be environment variables: the site wrapper
  parses them as text before bash runs. They are the only settings in that file
  which are not overridable, and they ship as placeholders (see below).

### Adapting this to your site

Four lines in `finetune_chunk_slurm.sh` and one in `docker/Dockerfile.cluster`
carry placeholders and **must be edited before first use**:

| placeholder | what it is | where to get it |
|---|---|---|
| `<cluster-registry>` | registry hostname the site's `sbatch` wrapper pulls from | your cluster's container registry (this site mirrors NGC there); `nvcr.io` works for the base image if you have NGC access directly |
| `<project>` | registry namespace you can push to | your registry account |
| `<user>` | filesystem account owning `/home/share/<user>` | `whoami` on the login node |

The real values for the runs reported here are internal to the cluster and are
deliberately not published: this repository is public, and an internal hostname
paired with a valid account identifier is attack surface that buys an outside
reader nothing. `finetune_chunk_slurm.sh` fails in preflight — through the same
`FAIL:` path and job-status sentinel as every other check — if the placeholders
are still unedited, rather than letting you discover it after a queue wait.

## Round 2 — 814 hours, and the first strictly comparable comparison

> Status: **run completed.** Chunk CER 0.1623 on the same 5,194 clips round 1
> was scored on, down from 0.1679.

Every quality comparison up to this point had a caveat attached: the MPS
baseline and the 300 h cluster run were scored on different val sets, so the
numbers were directional at best. Round 2 removes the caveat.

### Pinning the val set

`--pin-val-keys FILE` takes a list of manifest `key` values and makes the val
set **exactly** those clips, holding their speakers out of train. Round 2 was
built with round 1's 5,194 val keys pinned, so `val.jsonl` came out with the
identical sha256 (`a4e3167e…`) as round 1's. The two runs are scored on the same
bytes, and round 1's numbers can be quoted beside round 2's without re-running
anything.

This exists because the alternative had already failed twice. Growing a corpus
re-runs the speaker-disjoint split over more data, which produces a *different*
val set; and when we checked whether the older val could simply be re-scored,
**547 of its 772 clips (70.9 %) had landed in the newer training set**. Pinning
is the only way a generation-over-generation number means anything.

There is no pinned-plus-top-up mode, deliberately: a superset val is exactly the
incomparability the flag exists to remove.

### Corpus

58 archives across 56 studios, selected the same way (studio diversity first),
with a tie-break added: prefer archives absent from the teacher corpus used in
the transcript audit below, which raised the audit's usable share from 12 to 37
of 58.

| | clips | hours |
|---|---|---|
| train | 549,404 | 813.81 |
| val (pinned) | 5,194 | 7.48 |

The build ran in batches of 6 archives because the three intermediate forms do
not fit the quota together (archives 40 GB + extracted 43 GB + converted audio
115 GB against ~118 GB free). Each batch converts, then deletes its `.7z` and
its extracted `.ogg` while **keeping `raw/<stem>/index.json`** — tiny, and the
only thing the manifest builder needs from the raw tree. `--manifest-only` then
rebuilds the whole-corpus split from those index files plus the converted wavs,
without download, extraction or conversion, and without needing credentials.

### Results

Same geometry, same fp32/TF32-off measurement discipline, same 5,194 clips.

| model | chunk CER | full CER | gap |
|---|---|---|---|
| base (published) | 0.4024 | 0.1781 | 0.2243 |
| round 1 — 300 h, epoch 4 | 0.1679 | 0.1584 | 0.0095 |
| **round 2 — 814 h, epoch 3** | **0.1623** | **0.1518** | 0.0106 |
| round 2, epoch 4 | 0.1655 | 0.1571 | 0.0084 |
| round 2, epoch 2 | 0.1670 | 0.1753 | −0.0084 |

**2.7× the data bought 3.3 % relative on chunk CER** (0.1679 → 0.1623) and
4.2 % on full-attention CER (0.1584 → 0.1518). Both moved together, so the gain
is not being paid for out of offline quality. Against the base model the chunk
CER falls 59.7 %.

The honest reading is that returns are diminishing sharply. Round 1 took chunk
CER from 0.4024 to 0.1679 with 300 h; another 514 h moved it 0.0056 further.
Whatever is left is unlikely to be data volume.

Epoch 2 is an oddity worth recording: its gap is **negative** (−0.0084), chunk
decoding scoring *better* than full attention on the same clips. Nothing here
explains it, and it did not persist into later epochs.

**The 800-clip selection sweep mis-ranked the finalists for the third time.** It
put epoch 2 second (0.1556) and epoch 4 third (0.1567); the full set reverses
them and moves epoch 2 to last. Three independent failures across two rounds is
no longer bad luck — the subset separates good checkpoints from bad and cannot
resolve differences of a few thousandths. Run finalists on the whole set before
claiming a winner.

### Transcript audit

With a training run occupying two GPUs, the spare slot went to auditing the
ground truth rather than generating more of it. `scripts/detect_label_noise.py`
decodes clips with `litagin/anime-whisper` and ranks by CER against the stored
transcript — restricted to the 37 archives absent from that model's own training
corpus, since on the other 21 it has memorised the pairs and cannot detect an
error it was trained to reproduce.

Comparison happens on a **normalized projection of both strings**. The teacher
emits ASCII `!?`, no `。`, and single `…` where our corpus uses full-width `！？`,
`。` and `……`; measured over the real val, that convention gap alone is **8.29
CER points** raw and **0.000000** after normalization. Without it the filter
measures orthography, and because `……`/`！` density is highest on emotive lines,
it would preferentially flag those.

Three findings from 4,000 clips:

- **Unescaped newlines surviving as a literal `n`** (`一n杯`, `調n教`) — 89
  occurrences, concentrated in one title, ~12,000 clips extrapolated. A defect,
  and repairable: `normalize_text` now deletes a lone `n` with kana or kanji on
  both sides. Latin runs, `第N回` and doubled `nn` are pinned untouched by tests.
- **Titles transcribed entirely in kana** — 285 of 2,872 substantial transcripts
  contain no kanji at all, median CER 0.417 against 0.103 for the rest, and one
  title is 100 % kanji-free. Not wrong, but training on it teaches kana output.
  `--drop-kana-only-titles` excludes a title above a kanji-free fraction
  (default 0.8). Opt-in, and it drops nothing from the corpora on disk. Deliberately
  title-level: a short kana-only line is ordinary Japanese and dropping those
  would bias the corpus against backchannels.
- **One title that looked worst of all and is fine.** `hibiki_works_LOVELY_CATION`
  scored a median CER of 1.000. Its ground truth correlates with clip duration
  normally (r = 0.819 against controls' 0.844–0.914) at a normal 4.65
  characters/second; the *teacher* correlates at 0.327 and emits 1.34 chars/second,
  returning empty output on non-verbal and NSFW vocalisation. The transcripts are
  correct and the measuring instrument failed. Excluding it would have discarded
  good data on the strength of a broken measurement.

That last case generalises: **an empty teacher output is not evidence against a
transcript**, yet it currently scores as CER 1.0 and sorts to the top of the
report. Treating it as unscoreable is the obvious next fix.

### Two guards added after they were needed

Both come from failures that every existing check passed.

`EXPECT_TRAIN_HOURS` / `EXPECT_VAL_CLIPS` — a run trained to completion on
**16.5 hours** when 813.8 were intended, reported `SENSEVOICE_JOB_OK
preflight=passed`, and finished in 4 minutes instead of 90. A data-prep
invocation over a 4-archive subset had rebuilt the manifest and replaced the
full one; `prepare_vn_data.py` always rebuilds the split over the archives it is
given, which is by design and is exactly what makes it dangerous to run casually.
The job-status sentinel cannot see this — the run genuinely succeeded, at the
wrong thing. Declaring the expected size is the only check that can.

The val expectation is an exact match rather than a tolerance: the val is pinned
precisely so numbers stay comparable, and a silently replaced one would look
perfectly healthy while invalidating the comparison.

The second guard is a habit rather than code. A "full val" evaluation was
submitted with `LIMIT=` against a script reading `${LIMIT:-800}` — which treats
empty as unset — so it re-ran the 800-clip subset and wrote it under
`eval_full_*.json`. It was caught only because the numbers matched the selection
sweep to four decimals. **Assert `num_clips` before reading a CER**, and prefer
`${VAR-default}` when an empty value is meaningful.

## Round 3 — repairing the emotion head

> Status: **in progress.** Code landed and tested; corpus rebuild, labelling
> and training not yet run.

Rounds 1 and 2 trained the emotion head on a constant. `prepare_vn_data.py`
stamped `<|NEUTRAL|>` on every clip, and because the emotion slot shares the
CTC output projection, the head was optimised towards that constant.
`acc_rich` saturating at 1.0 by step 880 was the label being constant, not
the model being right.

Nothing caught it. CER is measured after `strip_rich_tags`, so every
evaluation this project has run was blind to the emotion output. The defect
survived two full training rounds and a published checkpoint.

### Saying "no label" without lying

The repair needs per-clip emotion labels, and pseudo-labels are not reliable
for every clip. The question is what to write for the rest.

`<|EMO_UNKNOWN|>` looks like the obvious answer and is the wrong one.
Inference bans it (`ban_emo_unk`), so training it is capacity spent on a
token that can never be emitted; and if most clips carried it, the run would
reproduce the original collapse with a different constant.

Instead the manifest carries `<|SER|>` — a single token, id 24991 — and
`model.py` maps it to `ignore_id` before the rich cross-entropy. That drops
the slot from `LabelSmoothingLoss`' numerator *and* denominator and blocks
the gradient through it, while the clip still trains CTC in full. The
mapping lives in the model because `prepare_vn_data.py` writes strings and
the funasr dataset layer tokenises them; neither can inject `-1`.

Two consequences worth knowing. The denominator only shrinks when
`length_normalized_loss` is true, which it is in the base config but is a
property of the configuration rather than of the mechanism. And masking
raises the effective weight of the other three rich slots by roughly 1.03 at
the expected mask rate, since they keep their share of a smaller denominator.

`acc_emo` reports the emotion slot's accuracy on its own. It exists because
its absence is what let round 2 fail silently.

### Two labellers, and only what they agree on

emotion2vec+ large reads the waveform; Qwen2.5-14B-Instruct reads the
ground-truth transcript. Where they agree the label is adopted, where they
disagree it is masked. Both are Apache-2.0 — the obvious Japanese
visual-novel SER resources are GPL-derived and would have tainted the result.

Agreement is worth the cost because this is acted dialogue: the audio side
hears exaggerated delivery, and the text side reads lines whose emotion is
carried entirely by the performance. Neither is trustworthy alone.

Text classification carries `embarrassed` and `sexual` as first-class
buckets. Both are frequent here and neither maps onto the seven SenseVoice
emotions, so giving them their own labels keeps them out of the seven. They
mask rather than drop: no clip leaves the corpus.

### How badly, measured against a corpus nobody here labelled

JVNV — 1,615 clips, four speakers, six emotions, human-labelled, CC BY-SA,
evaluation only. It contains **no neutral recordings at all**, so a
`<|NEUTRAL|>` prediction is wrong by construction.

| | accuracy | macro-F1 | dominant prediction | share |
|---|---|---|---|---|
| base, `--ban-emo-unk` | 0.318 | 0.261 | `<\|ANGRY\|>` | 63.7 % |
| round 2 epoch 3 | **0.000** | **0.000** | `<\|NEUTRAL\|>` | **99.75 %** |

Round 2 answers `<|NEUTRAL|>` for 1,611 of 1,615 clips and never once emits
any of the six target emotions. Accuracy is not low, it is exactly zero, and
the confusion matrix collapses every row into a single column. The result is
byte-identical with and without `--ban-emo-unk`, because the head never
reaches for `<|EMO_UNKNOWN|>` either.

The attribution is clean: **base emits zero `<|NEUTRAL|>` predictions**, so
this is not the architecture, it is what round 2's training did to it.

Base is a weak bar and should be read as one — it over-predicts `<|ANGRY|>`
on 63.7 % of clips and is blind to `<|DISGUSTED|>` entirely. It clears chance
(16.7 %) and the majority class (18.9 %), so it is a real floor, but not a
strong one.

Two measurement notes. Without `--ban-emo-unk`, base abstains on 98.1 % of
clips and scores 1.24 %, which makes the base-versus-round-2 gap read as a
floor effect rather than a real ordering — the banned run is the one that
answers the question. And the first run of this evaluation was submitted
with a `NameError` that fires *after* all decoding completes; on this
scheduler that surfaces as a COMPLETED job with an empty output file.

### Why round 3 still starts from round 2, and what would prove that wrong

Round 3 initialises from round 2 epoch 3 rather than from base. The reasoning
rests on a hypothesis that is **not established**, and is recorded here as a
hypothesis so it does not quietly become a fact:

> the emotion head's output layer is saturated towards a constant, while the
> encoder representations are intact.

What supports it: round 2 epoch 3 is the best ASR checkpoint measured
(chunk CER 0.1623), so the encoder still produces features good enough to
transcribe from; and the JVNV failure is perfectly uniform across all six
classes, which is the signature of an output-side constant rather than
degraded features. What would settle it properly is a linear probe on frozen
encoder features — not run, because the alternative is cheaper.

**The fallback is the experiment.** A base-initialised rerun costs about two
hours, so P5's JVNV measurement doubles as the test: if round 3's JVNV
macro-F1 comes in **below base's 0.261**, the run is repeated once from base
initialisation, and that rerun's result is what decides whether the encoder
was damaged after all. Reporting per-class F1 is required either way —
whether round 3 can score non-zero on `<|DISGUSTED|>`, where base is blind,
is qualitative evidence of having genuinely surpassed base rather than
having edged past its average.

### The corpus, and the pin that moved

801.10 h of train against round 2's 813.81. The arithmetic is worth writing
out, because neither term is what was expected:

```
  813.8127 h   round 2
-  12.7095 h   --drop-kana-only-titles (JADE_Love_Destination, 6,009 clips)
+   0.0000 h   stem-resolution recovery
= 801.1032 h   round 3, measured
```

**The ~90 hours the stem-resolution fix was meant to recover are not on
disk.** Round 2's `manifest.json` was itself built with `--manifest-only`, so
its `dropped_missing_audio: 57,180` already meant "index entry that matched no
wav **on disk**", not "no `.ogg` at extraction time". 54,420 of those belong to
four titles, two of which have no `audio/` directory at all. The wav count on
disk (563,268) equals the manifest's `kept` exactly, across all 58 titles with
zero per-title mismatches — every wav present was already found by the old
exact-path resolver, so the stem fallback has nothing left to match. The
rebuild confirmed it prospectively: `resolved_audio_stem_match_elsewhere: 0`.

The four `.7z` still on disk are exactly those four titles, and that is
causal rather than coincidental — the batch pipeline deletes an archive once
its conversion succeeds, so these survive *because* conversion never
completed. Recovering the hours means re-extracting and re-converting them.
**Deferred to a later round**: round 2 measured 2.7× the data buying 3.3 %
relative on chunk CER, so +11 % would buy under 0.2 % — below the evaluation's
resolution — and changing corpus size while repairing the emotion head would
make round 3 a two-variable experiment.

**`EXPECT_TRAIN_HOURS=801.10`.** Note that the guard could not have caught the
stale value: at round 2's 813.81 the ±5 % band is 773.12–854.50 h, and 801.10
sits inside it. A 1.56 % drift passes, so this update has to be made
deliberately.

**The pinned val changed by one record.** `val.jsonl` moved from
`a4e3167e56d09f954b46addb58a793d4dc9502f45e297a9ba29ef246038c190e` to
`c57be1c82d4df78af1a5116e220ac4f7b402aaa5dfb7ebbc0065a04a33194b68`.
Exactly 1 of 5,194 records differs, in `target`
and `target_len` only — same keys, same order, same `source`, same
`source_len`. The lone-`n` repair landed on a val transcript:

```
Libido_Soft_Hinekuremono_..._Reversible__太陽__taiyo0007
 old: …パーッと遊びにn行こうって話に…今日部活n休みなんだよ！  (57)
 new: …パーッと遊びに行こうって話に…今日部活休みなんだよ！    (55)
```

The repair was accepted. Keeping a knowingly-corrupt reference so a hash
matches would fix the defect into every future round's scoring, and the pin
exists to make numbers comparable, not to preserve errors. But the swap is
only legitimate if it is *shown* not to move the numbers, so round 2 epoch 3
is re-scored on the new val and both figures are recorded — two characters in
roughly 1.5 M should agree to four decimals, and if they do not, that is a
finding rather than a rounding note. **`c57be1c8…` is the pin from here on.**

### The audio labeller does not transfer to this domain

A 5,000-clip pilot settled a rule that had been left open on purpose. The
result is that **`--audio-conf-fallback` is retired permanently.**

emotion2vec+ large over visual-novel dialogue:

| class | share |
|---|---|
| surprised | 32.1 % |
| happy | 24.0 % |
| neutral | **1.64 %** |
| other + unknown (masked) | 3.90 % |

Japanese conversational speech is not 32 % surprised and 1.6 % neutral. The
labeller is reading acted delivery as constant emotional peak — the exact
domain shift the two-labeller design was built to survive, now measured
rather than anticipated.

**Confidence is not a proxy for correctness here.** About 85 % of clips score
above 0.7, and ordinary conversational lines come back as `surprised` at
score **1.000**. The retired rule was "if audio is confident, override a
text-neutral disagreement"; at this distribution that threshold selects
almost the whole corpus and would import the domain shift wholesale.

The disagreement confusion matrix stays in the stats. It is now more useful,
not less — it is how the shift is read off the full corpus, and it is the
input to whichever merge rule replaces the retired one.

One direction is still open and deliberately unbuilt: audio saying `neutral`
is rare, and in the pilot those calls landed on short flat utterances that
really were flat. A labeller can be worthless in one direction and
informative in the other. That is a hypothesis, it is unmeasured, and the
merge analysis is what will decide it.

**A test can only check the assumption it was written from.** The pilot's
first two submissions died in seconds on a smoke gate, because
`emotion2vec_plus_large`'s `tokens.txt` ends in a bare `<unk>` rather than the
`中文/english` pair its other eight classes use — and both the docstring and
the unit-test fixture asserted the pair. The fixture was written from an
assumption, so the suite checked the assumption against itself and passed. It
is now reconstructed from the real file and cross-checked against two
independent quantities of it, its length and its sha256, with an opt-in check
that reads the staged artefact directly.

Raising on an unknown class rather than defaulting to the mask is what made
this cost three seconds of GPU. A fallback would have exited cleanly over
550,000 clips having masked everything it did not recognise.

### The rule that was chosen, and what it actually does

Training labels come from **B1: text-led with an audio neutral veto**, NEUTRAL
cap 0.5. Measured yield on the pilot's 5,000 clips: 3,546 usable, **70.92 %**.
Agreement-only is kept for *evaluation* — the val consensus subset that SER
accuracy is scored against is built with it, which is why it remains the
default and why its output is pinned byte-identical by test.

**The veto does not do what its name says.** It was approved on the theory
that the audio labeller catches the text labeller's mistakes. Reading the
cases it actually fires on, it catches something narrower and more specific:
the text labeller's **semantic over-reading** — flat, level utterances that
merely *describe* an emotional situation, which the text side scores as the
emotion being described. That is a better justification than the one it was
approved on, and a different one. Recorded here so the original rationale
does not stand unchallenged.

**B1 is closer to single-labeller text labelling than the name suggests.**
The veto's ceiling is 1.64 % — the rate at which the audio labeller says
`neutral` at all — and it fired on 19 of 5,000 clips. Yield is set almost
entirely by text abstention (27.4 %) and the cap. The audio side holds a veto
it can exercise on roughly one clip in sixty.

Two known error modes, both left in deliberately:

- **Laughter reads as sadness.** Where audio says `happy` on a laugh and text
  reads the line as `sad`, B1 adopts `sad`. The veto cannot help: it fires
  only from `audio=neutral`, so a confident audio *disagreement* has no route
  to block anything. This is structural, not a tuning gap.
- **`sexual` false positives cost coverage.** `sexual` is 19.6 % of text
  labels and maps to the mask, so over-firing removes real supervision
  silently. Worth watching in the full-corpus distribution.

Neither is fixed by adjusting the merge. Both are carried by the acceptance
gates instead — JVNV macro-F1 against base, and consensus-subset SER accuracy
— which is where a bad labelling rule should surface. Choosing the gates as
the defence rather than tightening the rule is the deliberate decision here.

### A second image, because vLLM and numpy cannot share one

`docker/Dockerfile.textlabel` is separate from `Dockerfile.cluster` and will
stay separate. Every current vLLM requires `numpy>=2`; this project pins
`numpy<=1.26.4`, and the `scipy` / `numba` / `llvmlite` / `scikit-learn` /
`umap-learn` caps in `requirements.txt` exist to hold that ceiling. vLLM
would also move torch to whatever it was compiled against. The conflict is
not reconcilable, and it does not need to be: the text labeller reads a
manifest and writes a JSONL, sharing nothing with the training stack at
runtime.

The base is `nvidia/cuda:13.0.3-runtime-ubuntu24.04` rather than the NGC
mirror, matching vLLM 0.27.1's torch 2.13/cu13 wheels.

Qwen2.5-14B's weights are **staged, not baked in** — 28 GB of layer would be
pulled on every job, and the quota has about 30 GB of margin. Stage them and
pass the path via `--model`.

Two operational notes. Build on an **x86_64 host** with
`--platform linux/amd64`; under qemu on Apple Silicon the build-time
verification layers can fail as false positives, which is worse than not
having them. And if `--tensor-parallel-size` exceeds 1, the job needs
`#CONTAINER --shm-size 32G` — NCCL's shared-memory transport otherwise fails
in a way that reads as a hang rather than an error.

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

**Now verified:** that a chunk finetune recovers quality. A checkpoint exists
and closes 90% of the chunk-versus-full CER gap on held-out Japanese — see
"The Japanese chunk finetune" above. The caveat on the deviation magnitudes
below still stands, since those come from a small random encoder rather than
this checkpoint.

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
