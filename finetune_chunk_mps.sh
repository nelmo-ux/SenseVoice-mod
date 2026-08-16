#!/usr/bin/env bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
#
# Dynamic chunk-mask finetuning for SenseVoiceSmall on Apple Silicon (MPS).
#
# This is the MPS sibling of finetune_chunk.sh.  It follows that script's SMOKE
# path -- funasr's bin/train.py -- not its GPU path.  The GPU path cannot be
# retargeted: funasr/bin/train_ds.py unconditionally executes
#     kwargs["device"] = int(os.environ.get("LOCAL_RANK", 0))
#     trainer.device   = int(os.environ.get("LOCAL_RANK", 0))
# right after warp_model, so every batch is sent to *accelerator index 0*
# regardless of ++device.  bin/train.py instead takes its device from the model
# itself (``kwargs["device"] = next(model.parameters()).device``) after an
# honest ``model.to(device)``, which is what makes ++device=mps work at all.
#
# Everything else -- model, dataset, sampler, chunk config -- is shared between
# the two trainers, so this runs the same computation the GPU path would.
#
# Usage:
#   ./finetune_chunk_mps.sh                  # run the finetune (long: ~6 h)
#   ./finetune_chunk_mps.sh --dry-run        # preflight + print resolved command
#   MAX_EPOCH=1 ./finetune_chunk_mps.sh      # override any setting below
#
# Every setting is an environment variable with a default; see CONFIGURATION.

set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --dry-run    Run every preflight check, report failures as warnings instead
               of aborting, print the fully resolved command, and exit without
               launching training.
  -h, --help   This message.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Error: unknown argument: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
soft_failures=0

info() { printf '%s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }

# Fatal no matter what: without these the command cannot even be resolved.
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

# Fatal for a real run, downgraded to a warning under --dry-run so that the
# resolved command can still be inspected before the data exists.
check_fail() {
    if [ "${DRY_RUN}" = "1" ]; then
        warn "$*"
        soft_failures=$((soft_failures + 1))
    else
        printf 'Error: %s\n' "$*" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# CONFIGURATION -- every value below is overridable from the environment
# ---------------------------------------------------------------------------

# --- paths ---
MODEL_DIR="${MODEL_DIR:-${workspace}/models/SenseVoiceSmall}"
TRAIN_JSONL="${TRAIN_JSONL:-${workspace}/data/vn/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${workspace}/data/vn/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${workspace}/outputs/chunk_mps}"

# --- schedule ---
# 4 epochs over ~53.8 h of audio.  At the measured ~36 audio-hours per wall-hour
# (batch 12, fp32, MPS) that is 53.8/36 ~= 1.5 h per epoch, ~6 h total.
MAX_EPOCH="${MAX_EPOCH:-4}"

# --- learning rate ---
# 2e-4 is a CEILING, inherited from finetune_chunk.sh.  It must NEVER be raised:
# this is an adaptation of an already-trained checkpoint, and the failure mode
# to fear is catastrophic forgetting of full-attention behaviour, not
# underfitting.  If the per-epoch evaluation shows forgetting, LOWER this (5e-5
# or 1e-5 are the natural next steps) and re-run.  The check below refuses to
# start if it is raised.
LR_CEILING="0.0002"
LR="${LR:-0.0002}"

# Left empty on purpose: inherit scheduler_conf.warmup_steps from the base
# model's config.yaml (25000).  That is far more than this run's total steps, so
# warmuplr never leaves its linear ramp and the LR actually reached stays well
# below the ceiling.  The preflight prints the projected peak so this is visible
# rather than surprising.  Set to override.
WARMUP_STEPS="${WARMUP_STEPS:-}"

# --- batching: MEMORY is the binding constraint on MPS, not compute ---
# Measured MPS driver allocation on this machine (fp32):
#     batch  8 -> 26 GB
#     batch 12 -> 35 GB
#     batch 24 -> 69 GB   <-- past the 55.7 GB recommended cap; 2.7-5.4x slower
# So the useful range is batch 8-12, i.e. roughly 72 audio-seconds per step.
#
# We cap by AUDIO DURATION rather than by funasr's GPU-oriented token budget
# (batch_size=6000).  Chunk masking expands the sequence before folding it back,
# so a token count is a poor proxy for the peak activation footprint, whereas
# total audio in the batch tracks it directly.
#
# Units: dataset_conf.batch_size is compared against the sampler's
# ``max_len_in_batch * len(batch)``, which is built from the jsonl's
# ``source_len`` field.  This repo writes source_len in 10 ms frames (see
# scripts/make_smoke_data.py: ``clip.shape[0] / samplerate * 1000 / FRAME_MS``,
# FRAME_MS=10), matching the shipped data/train_example.jsonl.  Hence
# 100 units per audio-second.  The preflight verifies this against the real
# audio headers and refuses to start if the corpus uses another convention --
# getting it wrong would move the memory footprint by 6x.
SOURCE_LEN_UNITS_PER_SECOND=100
MAX_AUDIO_SECONDS_PER_STEP="${MAX_AUDIO_SECONDS_PER_STEP:-72}"
BATCH_TOKENS="${BATCH_TOKENS:-$((MAX_AUDIO_SECONDS_PER_STEP * SOURCE_LEN_UNITS_PER_SECOND))}"
# Hard ceiling on examples per batch (funasr default is 200).  The duration cap
# above is the primary lever; this one stops a bucket of very short clips from
# packing an unreasonable number of examples into one step.
MAX_SAMPLES_PER_STEP="${MAX_SAMPLES_PER_STEP:-12}"
# Length-bucketing window.  Larger = tighter length grouping = less padding =
# less memory, at the cost of shuffling diversity.  Matches the GPU path.
SORT_SIZE="${SORT_SIZE:-1024}"
# Dataloader workers only do audio load + fbank on CPU tensors; they never touch
# MPS.  Set to 0 if the loader ever hangs on this machine.
NUM_WORKERS="${NUM_WORKERS:-2}"
ACCUM_GRAD="${ACCUM_GRAD:-1}"

# --- checkpointing ---
# funasr's bin/train.py saves ``model.pt.ep{N}`` at the end of every epoch
# unconditionally, so per-epoch checkpoints come for free.  What is NOT free is
# keeping them: Trainer.save_checkpoint prunes ``saved_ckpts`` down to
# keep_nbest_models and os.remove()s the loser.  A large value disables that in
# practice, so all four epochs survive for the forgetting evaluation.
# (keep_nbest_models=0 also disables pruning today -- ``if self.keep_nbest_models
# > 0`` -- but 0 reads like "keep none", so prefer an unambiguous large value.)
KEEP_NBEST="${KEEP_NBEST:-1000}"
# Final checkpoint averaging writes an extra file; 1 makes it a no-op. The point
# of this run is to pick a single epoch by evaluation, not to average.
AVG_NBEST="${AVG_NBEST:-1}"
# Mid-epoch safety checkpoints, so a crash at hour 5 does not cost a whole
# epoch.  funasr asserts save_checkpoint_interval == validate_interval, so this
# single knob drives both.  These are extra; they do not replace the per-epoch
# saves and are also kept.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
# Resume from OUTPUT_DIR/model.pt if it exists.
RESUME="${RESUME:-true}"

# --- logging ---
LOG_INTERVAL="${LOG_INTERVAL:-20}"

# --- device ---
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-0}"

# --- chunk geometry (must match finetune_chunk.sh exactly) ---
# See docs/chunk_training.md: chunk_size is the TOTAL window width
# (pad_left + stride + pad_right), which is NOT the inference-side convention.
# Each entry is one dynamic-chunk operating point; one is drawn per step.
# An entry <= 0 is a full-attention sentinel and skips the geometry checks.
#
# This geometry is TRIPLICATED across the repo -- here, in finetune_chunk.sh,
# and in scripts/eval_chunk_gap.py's TRAINING_CHUNK_CONFIG -- and nothing links
# the three.  The DEFAULT_* constants exist so that at least this file cannot
# drift from itself: they are both the defaults applied below and the reference
# the drift warning compares against.  Overriding any of the four variables
# makes this run disagree with the evaluator; see the warning further down.
DEFAULT_CHUNK_SIZES="8,12,16"
DEFAULT_STRIDES="6,10,14"
DEFAULT_PAD_LEFTS="0,0,0"
DEFAULT_LOOK_BACKS="1,1,1"

CHUNK_SIZES="${CHUNK_SIZES:-${DEFAULT_CHUNK_SIZES}}"
STRIDES="${STRIDES:-${DEFAULT_STRIDES}}"
PAD_LEFTS="${PAD_LEFTS:-${DEFAULT_PAD_LEFTS}}"
LOOK_BACKS="${LOOK_BACKS:-${DEFAULT_LOOK_BACKS}}"

# --- optional MPS allocator guardrail ---
# Unset by default.  Set to 1.0 to make the allocator hard-fail once it passes
# torch.mps.recommended_max_memory() instead of silently degrading throughput.
MPS_HIGH_WATERMARK_RATIO="${MPS_HIGH_WATERMARK_RATIO:-}"

# ---------------------------------------------------------------------------
# Learning-rate ceiling
# ---------------------------------------------------------------------------
if awk -v a="${LR}" -v b="${LR_CEILING}" 'BEGIN { exit !(a + 0 > b + 0) }'; then
    die "LR=${LR} exceeds the ceiling ${LR_CEILING}. Lowering it is fine; raising it is not."
fi

# ---------------------------------------------------------------------------
# Chunk geometry validation -- fail before anything expensive starts
# ---------------------------------------------------------------------------
# Hydra parses "[a,b,c]" as a list, but bare brackets are a glob pattern to the
# shell, so the bracketed forms must stay quoted all the way to the trainer.
# The scalars above are the single source of truth; the hydra overrides are
# built from them, so validation cannot drift from what is actually passed.
list_to_words() {
    local raw="$1"
    raw="${raw#[}"
    raw="${raw%]}"
    printf '%s' "${raw//,/ }"
}

is_int() {
    case "$1" in
        ''|*[!0-9-]*) return 1 ;;
        *) return 0 ;;
    esac
}

validate_chunk_geometry() {
    local cs st pl lb
    read -r -a cs <<<"$(list_to_words "${CHUNK_SIZES}")"
    read -r -a st <<<"$(list_to_words "${STRIDES}")"
    read -r -a pl <<<"$(list_to_words "${PAD_LEFTS}")"
    read -r -a lb <<<"$(list_to_words "${LOOK_BACKS}")"

    local n=${#cs[@]}
    [ "${n}" -gt 0 ] || die "CHUNK_SIZES is empty"
    [ "${#st[@]}" -eq "${n}" ] || die "STRIDES has ${#st[@]} entries, CHUNK_SIZES has ${n}"
    [ "${#pl[@]}" -eq "${n}" ] || die "PAD_LEFTS has ${#pl[@]} entries, CHUNK_SIZES has ${n}"
    [ "${#lb[@]}" -eq "${n}" ] || die "LOOK_BACKS has ${#lb[@]} entries, CHUNK_SIZES has ${n}"

    local i c s p l pr
    for (( i = 0; i < n; i++ )); do
        c="${cs[$i]}"; s="${st[$i]}"; p="${pl[$i]}"; l="${lb[$i]}"
        for v in "${c}" "${s}" "${p}" "${l}"; do
            is_int "${v}" || die "chunk entry ${i}: '${v}' is not an integer"
        done

        if [ "${c}" -le 0 ]; then
            info "  entry ${i}: chunk_size=${c} -> full-attention sentinel (geometry checks skipped)"
            continue
        fi

        [ "${s}" -gt 0 ] \
            || die "chunk entry ${i}: stride=${s} violates 0 < stride"
        [ "${s}" -le "${c}" ] \
            || die "chunk entry ${i}: stride=${s} > chunk_size=${c} violates stride <= chunk_size"
        [ "${p}" -ge 0 ] \
            || die "chunk entry ${i}: pad_left=${p} violates 0 <= pad_left"
        [ "${p}" -le $(( c - s )) ] \
            || die "chunk entry ${i}: pad_left=${p} > chunk_size-stride=$(( c - s )) violates pad_left <= chunk_size - stride"

        pr=$(( c - s - p ))
        if [ "${l}" -ne 0 ] && [ "${pr}" -lt 1 ]; then
            die "chunk entry ${i}: look_back=${l} != 0 requires pad_right >= 1, but pad_right=${pr}. The attention cache is built by dropping the last pad_right frames and would be empty."
        fi
        if [ "${l}" -ne 0 ] && [ "${p}" -gt 0 ]; then
            warn "chunk entry ${i}: look_back != 0 with pad_left=${p} > 0 re-appends left-context frames to the attention cache (upstream funasr behaviour). Prefer pad_left=0 with look-back."
        fi

        # One encoder frame is 60 ms (10 ms mel hop x LFR factor 6).
        info "  entry ${i}: chunk_size=${c} stride=${s} pad_left=${p} pad_right=${pr} look_back=${l} -> window $(( c * 60 )) ms, commit $(( s * 60 )) ms, lookahead $(( pr * 60 )) ms"
    done
}

# Accept both "8,12,16" and "[8,12,16]" from the environment, then normalise, so
# the hydra overrides below cannot end up double-bracketed.
CHUNK_SIZES="$(list_to_words "${CHUNK_SIZES}" | tr -s ' ' ',')"
STRIDES="$(list_to_words "${STRIDES}" | tr -s ' ' ',')"
PAD_LEFTS="$(list_to_words "${PAD_LEFTS}" | tr -s ' ' ',')"
LOOK_BACKS="$(list_to_words "${LOOK_BACKS}" | tr -s ' ' ',')"

info "chunk geometry:"
validate_chunk_geometry

# ---------------------------------------------------------------------------
# Chunk geometry drift against the evaluator
# ---------------------------------------------------------------------------
# scripts/eval_chunk_gap.py hardcodes this geometry in TRAINING_CHUNK_CONFIG and
# its only knob, --geometry-index, selects one of those hardcoded entries -- it
# cannot be handed a different geometry from the command line.  So an override
# here does not merely go unnoticed by the evaluator: the evaluator will decode
# a geometry the model was never trained on and report the resulting CER as if
# it meant something.  Warn loudly rather than fail: a deliberate geometry sweep
# is a legitimate thing to want, it just cannot be evaluated without a matching
# edit on the other side.
GEOMETRY_IS_DEFAULT=1
if [ "${CHUNK_SIZES}" != "${DEFAULT_CHUNK_SIZES}" ] \
    || [ "${STRIDES}" != "${DEFAULT_STRIDES}" ] \
    || [ "${PAD_LEFTS}" != "${DEFAULT_PAD_LEFTS}" ] \
    || [ "${LOOK_BACKS}" != "${DEFAULT_LOOK_BACKS}" ]; then
    GEOMETRY_IS_DEFAULT=0
    cat >&2 <<EOF

***************************************************************************
*** WARNING: NON-DEFAULT CHUNK GEOMETRY -- EVALUATION WILL BE INVALID    ***
***************************************************************************
  this run trains:  chunk_size=[${CHUNK_SIZES}] stride=[${STRIDES}]
                    pad_left=[${PAD_LEFTS}] look_back=[${LOOK_BACKS}]
  the default is:   chunk_size=[${DEFAULT_CHUNK_SIZES}] stride=[${DEFAULT_STRIDES}]
                    pad_left=[${DEFAULT_PAD_LEFTS}] look_back=[${DEFAULT_LOOK_BACKS}]

  scripts/eval_chunk_gap.py HARDCODES the default geometry in its
  TRAINING_CHUNK_CONFIG constant, and its --geometry-index flag only chooses
  among those hardcoded entries -- there is no command-line way to give it the
  geometry above.  Run unchanged against this checkpoint it will decode with a
  geometry the model never trained on, and every CER/WER number it prints will
  be measuring that mismatch rather than the finetune.

  Before evaluating, edit TRAINING_CHUNK_CONFIG in scripts/eval_chunk_gap.py to
  match the values above (the same four lists are also hardcoded a third time
  in finetune_chunk.sh).  The resolved geometry is recorded in
  \${OUTPUT_DIR}/chunk_geometry.json for exactly this purpose.
***************************************************************************

EOF
fi

CHUNK_ARGS=(
    "++encoder_conf.chunk_size=[${CHUNK_SIZES}]"
    "++encoder_conf.stride=[${STRIDES}]"
    "++encoder_conf.pad_left=[${PAD_LEFTS}]"
    "++encoder_conf.encoder_att_look_back_factor=[${LOOK_BACKS}]"
)

# ---------------------------------------------------------------------------
# Chunk geometry record
# ---------------------------------------------------------------------------
# The geometry a checkpoint was trained with is otherwise unrecoverable from the
# checkpoint itself, which leaves evaluation and any later forensics guessing
# from whatever the three hardcoded copies happen to say today.  Write it down
# next to the checkpoints instead.
json_string() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "${s}"
}

# pad_right is implied (chunk_size - stride - pad_left) and is the number the
# lookahead latency actually depends on, so record it rather than making every
# reader re-derive it.  Full-attention sentinels have no pad_right -> null.
chunk_pad_rights() {
    local cs st pl i c s p out=""
    read -r -a cs <<<"$(list_to_words "${CHUNK_SIZES}")"
    read -r -a st <<<"$(list_to_words "${STRIDES}")"
    read -r -a pl <<<"$(list_to_words "${PAD_LEFTS}")"
    for (( i = 0; i < ${#cs[@]}; i++ )); do
        c="${cs[$i]}"; s="${st[$i]}"; p="${pl[$i]}"
        if [ "${c}" -le 0 ]; then
            out="${out}${out:+,}null"
        else
            out="${out}${out:+,}$(( c - s - p ))"
        fi
    done
    printf '%s' "${out}"
}

write_chunk_geometry_json() {
    local dest="$1"
    local tmp="${dest}.tmp.$$"
    local is_default="false"
    [ "${GEOMETRY_IS_DEFAULT}" = "1" ] && is_default="true"

    # Written to a temp file and renamed so a reader racing the launch sees
    # either the previous run's record or this one's, never a half-written file.
    cat >"${tmp}" <<EOF
{
  "schema_version": 1,
  "written_by": "finetune_chunk_mps.sh",
  "written_at": $(json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "pid": $$,
  "output_dir": $(json_string "${OUTPUT_DIR}"),
  "model_dir": $(json_string "${MODEL_DIR}"),
  "device": $(json_string "${DEVICE}"),
  "lr": $(awk -v x="${LR}" 'BEGIN { printf "%.10g", x + 0 }'),
  "max_epoch": $(json_string "${MAX_EPOCH}"),
  "seed": $(json_string "${SEED}"),
  "is_default_geometry": ${is_default},
  "encoder_conf": {
    "chunk_size": [${CHUNK_SIZES}],
    "stride": [${STRIDES}],
    "pad_left": [${PAD_LEFTS}],
    "encoder_att_look_back_factor": [${LOOK_BACKS}]
  },
  "derived": {
    "pad_right": [$(chunk_pad_rights)],
    "encoder_frame_ms": 60
  },
  "default_geometry": {
    "chunk_size": [${DEFAULT_CHUNK_SIZES}],
    "stride": [${DEFAULT_STRIDES}],
    "pad_left": [${DEFAULT_PAD_LEFTS}],
    "encoder_att_look_back_factor": [${DEFAULT_LOOK_BACKS}]
  },
  "eval_note": "scripts/eval_chunk_gap.py hardcodes TRAINING_CHUNK_CONFIG and its --geometry-index only selects among those hardcoded entries; if is_default_geometry is false, edit that constant to encoder_conf above before trusting any evaluation of this checkpoint"
}
EOF
    mv "${tmp}" "${dest}"
}

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${workspace}/.venv/bin/python" ]; then
        PYTHON_BIN="${workspace}/.venv/bin/python"
    elif PYTHON_BIN="$(command -v python3 2>/dev/null)"; then
        :
    else
        die "no python3 found and ${workspace}/.venv/bin/python does not exist"
    fi
fi
[ -x "${PYTHON_BIN}" ] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
info "python: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# funasr trainer entry point
# ---------------------------------------------------------------------------
# Deliberately bin/train.py, not bin/train_ds.py -- see the header.
find_train_tool() {
    local funasr_bin funasr_pkg candidate
    if funasr_bin="$(command -v funasr 2>/dev/null)"; then
        candidate="$(dirname "${funasr_bin}")/train.py"
        [ -f "${candidate}" ] && { printf '%s' "${candidate}"; return 0; }
    fi
    if funasr_pkg="$("${PYTHON_BIN}" -c 'import funasr, os; print(os.path.dirname(funasr.__file__))' 2>/dev/null)"; then
        candidate="${funasr_pkg}/bin/train.py"
        [ -f "${candidate}" ] && { printf '%s' "${candidate}"; return 0; }
    fi
    return 1
}

train_tool="$(find_train_tool)" \
    || die "funasr's bin/train.py not found. Is funasr installed in ${PYTHON_BIN}?"
info "trainer: ${train_tool}"

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
# A local directory is required rather than a hub id: funasr pip-installs the
# requirements.txt bundled inside a downloaded model directory, which can
# override this repository's dependency pins.  models/ is gitignored.
if [ ! -d "${MODEL_DIR}" ]; then
    check_fail "base model directory not found: ${MODEL_DIR} (see README 'Using a local model directory')"
elif [ ! -f "${MODEL_DIR}/model.pt" ] || [ ! -f "${MODEL_DIR}/config.yaml" ]; then
    check_fail "base model directory ${MODEL_DIR} is missing model.pt and/or config.yaml"
fi
info "model: ${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
for data_file in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
    if [ ! -f "${data_file}" ]; then
        check_fail "data manifest not found: ${data_file} (generate the VN corpus first; data/vn/ is gitignored)"
    elif [ ! -s "${data_file}" ]; then
        check_fail "data manifest is empty: ${data_file}"
    fi
done

# ---------------------------------------------------------------------------
# Preflight: MPS, corpus sanity, projected schedule
# ---------------------------------------------------------------------------
# Everything that needs to look inside torch or the corpus lives here, in one
# subprocess, so a bad environment is reported before six hours are spent on it.
run_preflight() {
    PYTORCH_ENABLE_MPS_FALLBACK=1 "${PYTHON_BIN}" - \
        "${DEVICE}" \
        "${TRAIN_JSONL}" \
        "${VAL_JSONL}" \
        "${MODEL_DIR}" \
        "${MAX_EPOCH}" \
        "${LR}" \
        "${WARMUP_STEPS}" \
        "${MAX_AUDIO_SECONDS_PER_STEP}" \
        "${MAX_SAMPLES_PER_STEP}" \
        "${SOURCE_LEN_UNITS_PER_SECOND}" \
        "${BATCH_TOKENS}" \
        <<'PREFLIGHT_PY'
import json
import math
import os
import sys

(
    device, train_jsonl, val_jsonl, model_dir, max_epoch, lr, warmup_arg,
    sec_cap, samp_cap, units_per_sec, batch_tokens,
) = sys.argv[1:12]

max_epoch = int(max_epoch)
lr = float(lr)
sec_cap = float(sec_cap)
samp_cap = int(samp_cap)
units_per_sec = float(units_per_sec)
batch_tokens = int(batch_tokens)

errors = []
notes = []


def fail(msg):
    errors.append(msg)


# --- torch / MPS -----------------------------------------------------------
try:
    import torch
except Exception as exc:  # pragma: no cover - environment problem
    print(f"  torch: NOT IMPORTABLE ({exc})")
    sys.exit(1)

print(f"  torch: {torch.__version__}")
if device.startswith("mps"):
    built = torch.backends.mps.is_built()
    avail = torch.backends.mps.is_available()
    print(f"  mps: built={built} available={avail}")
    if not built:
        fail("this torch build has no MPS backend")
    elif not avail:
        fail("MPS is not available (needs macOS 12.3+ on Apple Silicon)")
    else:
        rec = torch.mps.recommended_max_memory() / 1024**3
        print(f"  mps recommended_max_memory: {rec:.1f} GB")
        notes.append(
            f"stay under {rec:.1f} GB: batch 24 measured 69 GB and ran 2.7-5.4x slower"
        )

try:
    import funasr

    print(f"  funasr: {funasr.__version__}")
except Exception as exc:
    fail(f"funasr not importable: {exc}")

# --- warmup / projected peak LR -------------------------------------------
warmup = None
if warmup_arg.strip():
    warmup = float(warmup_arg)
else:
    cfg = os.path.join(model_dir, "config.yaml")
    try:
        import yaml

        with open(cfg, encoding="utf-8") as handle:
            warmup = float(yaml.safe_load(handle)["scheduler_conf"]["warmup_steps"])
        print(f"  warmup_steps: {warmup:.0f} (inherited from {cfg})")
    except Exception as exc:
        notes.append(f"could not read warmup_steps from {cfg}: {exc}")

# --- corpus ----------------------------------------------------------------
def inspect(path, label):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        fail(f"{label} manifest missing or empty: {path}")
        return None

    records, bad = [], 0
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad += 1
                if bad == 1:
                    fail(f"{label}: line {lineno} is not valid JSON")
    if not records:
        fail(f"{label}: no records in {path}")
        return None

    missing_keys = [
        k for k in ("source", "target", "source_len") if k not in records[0]
    ]
    if missing_keys:
        fail(f"{label}: records lack required key(s): {', '.join(missing_keys)}")
        return None

    lens = [int(r["source_len"]) for r in records]
    seconds = [n / units_per_sec for n in lens]
    total_h = sum(seconds) / 3600.0
    print(
        f"  {label}: {len(records)} clips, {total_h:.2f} h, "
        f"clip {min(seconds):.1f}-{max(seconds):.1f} s "
        f"(mean {sum(seconds) / len(seconds):.1f} s)"
    )

    # funasr silently drops clips outside these windows -- IndexDSJsonlRankFull
    # (max_source_length=2048, max_token_length=2200) and the sampler
    # (max_token_length=2048).  Report it so the real training hours are known.
    dropped = sum(1 for n in lens if n > 2048)
    if dropped:
        pct = 100.0 * dropped / len(records)
        notes.append(
            f"{label}: {dropped} clips ({pct:.1f}%) exceed source_len 2048 "
            f"(~{2048 / units_per_sec:.1f} s) and will be SILENTLY DROPPED by funasr; "
            "raise dataset_conf.max_source_length/max_token_length or re-segment"
        )

    # The 7z-derived audio must actually be on disk.  Sample rather than stat
    # tens of thousands of files.
    step = max(1, len(records) // 20)
    sample = records[::step][:20]
    absent = [r["source"] for r in sample if not os.path.isfile(r["source"])]
    if absent:
        fail(
            f"{label}: {len(absent)}/{len(sample)} sampled audio files do not exist "
            f"(e.g. {absent[0]}); extract the 7z corpus before training"
        )
    else:
        empty = [r["source"] for r in sample if os.path.getsize(r["source"]) == 0]
        if empty:
            fail(f"{label}: sampled audio file is zero bytes: {empty[0]}")
        else:
            print(f"  {label}: {len(sample)}/{len(sample)} sampled audio files present")

    # Verify the source_len convention against real audio headers.  Getting this
    # wrong moves the memory footprint by 6x, so it is a hard failure.
    try:
        import soundfile as sf

        rates = []
        for rec in sample[:8]:
            dur = sf.info(rec["source"]).duration
            if dur > 0:
                rates.append(int(rec["source_len"]) / dur)
        if rates:
            rates.sort()
            median = rates[len(rates) // 2]
            print(f"  {label}: source_len convention ~{median:.1f} units/second")
            if not 0.8 * units_per_sec <= median <= 1.25 * units_per_sec:
                fail(
                    f"{label}: source_len is ~{median:.1f} units/second but the batch "
                    f"budget assumes {units_per_sec:.0f} (10 ms frames). "
                    f"Set BATCH_TOKENS explicitly (= {sec_cap:.0f} x {median:.1f} "
                    f"~= {sec_cap * median:.0f}) or fix the manifest."
                )
    except ImportError:
        notes.append("soundfile unavailable: source_len convention not verified")
    except Exception as exc:
        notes.append(f"{label}: could not verify source_len convention: {exc}")

    return len(records), sum(seconds)


train = inspect(train_jsonl, "train")
inspect(val_jsonl, "val")

# --- projected schedule ----------------------------------------------------
if train:
    n_clips, total_seconds = train
    # Both caps bind; whichever forces more batches wins.
    steps_per_epoch = max(
        math.ceil(total_seconds / sec_cap),
        math.ceil(n_clips / samp_cap),
    )
    total_steps = steps_per_epoch * max_epoch
    print(
        f"  projected: ~{steps_per_epoch} steps/epoch, ~{total_steps} steps total "
        f"(cap {sec_cap:.0f} audio-s and {samp_cap} clips per step, "
        f"batch_size={batch_tokens})"
    )
    # Measured: ~36 audio-hours per wall-hour at batch 12, fp32.
    print(
        f"  projected: ~{total_seconds / 3600.0 * max_epoch / 36.0:.1f} h wall clock "
        f"at the measured 36 audio-h/wall-h"
    )
    if warmup:
        peak = lr * min(1.0, total_steps / warmup)
        print(
            f"  projected: peak LR ~{peak:.2e} "
            f"(warmuplr ramps lr*step/warmup; {total_steps} steps vs warmup {warmup:.0f})"
        )
        if total_steps < warmup:
            notes.append(
                f"the run ends inside warmup, so LR only reaches {peak:.2e} of the "
                f"{lr:.0e} ceiling and never decays. Lower WARMUP_STEPS to plateau."
            )

for note in notes:
    print(f"  note: {note}")
for err in errors:
    print(f"  FAIL: {err}")

sys.exit(1 if errors else 0)
PREFLIGHT_PY
}

info "preflight:"
if ! run_preflight; then
    check_fail "preflight checks failed (see FAIL lines above)"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_ASSIGNMENTS=()

# REQUIRED.  Without it training dies immediately with
#   NotImplementedError: aten::_ctc_loss is not implemented for the MPS device
# CTC loss has no MPS kernel, so it falls back to CPU.  Measured: numerically
# correct (relative difference 1e-7 against the CPU reference) and ~2% of step
# time.  This is not a workaround to remove later -- it is the only way the CTC
# head runs on this backend.
export PYTORCH_ENABLE_MPS_FALLBACK=1
ENV_ASSIGNMENTS+=("PYTORCH_ENABLE_MPS_FALLBACK=1")

# Without this the log stalls in the pipe to tee and a background monitor sees
# nothing for minutes at a time.
export PYTHONUNBUFFERED=1
ENV_ASSIGNMENTS+=("PYTHONUNBUFFERED=1")

# There is no CUDA here; make sure nothing tries.
export CUDA_VISIBLE_DEVICES=""
ENV_ASSIGNMENTS+=('CUDA_VISIBLE_DEVICES=""')

if [ -n "${MPS_HIGH_WATERMARK_RATIO}" ]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${MPS_HIGH_WATERMARK_RATIO}"
    ENV_ASSIGNMENTS+=("PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_HIGH_WATERMARK_RATIO}")
fi

# ---------------------------------------------------------------------------
# Trainer arguments
# ---------------------------------------------------------------------------
DATA_ARGS=(
    "++dataset_conf.batch_type=token"
    "++dataset_conf.batch_size=${BATCH_TOKENS}"
    "++dataset_conf.batch_size_sample_max=${MAX_SAMPLES_PER_STEP}"
    "++dataset_conf.sort_size=${SORT_SIZE}"
    "++dataset_conf.num_workers=${NUM_WORKERS}"
    # pin_memory is a CUDA concept; funasr defaults it to true and torch then
    # warns on every epoch that no accelerator was found for pinning.
    "++dataset_conf.pin_memory=false"
)

TRAIN_ARGS=(
    # bin/train.py defaults device to the literal string "cuda"
    # (``device = kwargs.get("device", "cuda")``), and an empty
    # CUDA_VISIBLE_DEVICES does not change that string -- torch would assert
    # "Torch not compiled with CUDA enabled".  The device has to be named.
    "++device=${DEVICE}"
    # Only consulted if someone launches with WORLD_SIZE>1; nccl does not exist
    # on this platform.
    "++backend=gloo"
    "++seed=${SEED}"
    "++train_conf.max_epoch=${MAX_EPOCH}"
    "++train_conf.log_interval=${LOG_INTERVAL}"
    "++train_conf.resume=${RESUME}"
    "++train_conf.accum_grad=${ACCUM_GRAD}"
    # funasr asserts these two are equal.
    "++train_conf.validate_interval=${SAVE_INTERVAL}"
    "++train_conf.save_checkpoint_interval=${SAVE_INTERVAL}"
    "++train_conf.keep_nbest_models=${KEEP_NBEST}"
    "++train_conf.avg_nbest_model=${AVG_NBEST}"
    # fp32.  torch.cuda.amp.GradScaler is CUDA-only and the 36 audio-h/wall-h
    # throughput figure was measured in fp32.
    "++train_conf.use_fp16=false"
    "++train_conf.use_bf16=false"
    "++train_conf.use_deepspeed=false"
)

if [ -n "${WARMUP_STEPS}" ]; then
    TRAIN_ARGS+=("++scheduler_conf.warmup_steps=${WARMUP_STEPS}")
fi

log_file="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
# Keep hydra's own run directory beside the checkpoints instead of scattering
# outputs/<date>/<time>/ trees at the repo root.
hydra_run_dir="${OUTPUT_DIR}/hydra/$(date +%Y%m%d_%H%M%S)"

CMD=(
    "${PYTHON_BIN}"
    "${train_tool}"
    "++model=${MODEL_DIR}"
    "++trust_remote_code=true"
    "++train_data_set_list=${TRAIN_JSONL}"
    "++valid_data_set_list=${VAL_JSONL}"
    "++dataset_conf.data_split_num=1"
    "++dataset_conf.batch_sampler=BatchSampler"
    "${DATA_ARGS[@]}"
    "${TRAIN_ARGS[@]}"
    "${CHUNK_ARGS[@]}"
    "++optim_conf.lr=${LR}"
    "++output_dir=${OUTPUT_DIR}"
    "hydra.run.dir=${hydra_run_dir}"
)

# ---------------------------------------------------------------------------
# OUTPUT_DIR concurrency lock
# ---------------------------------------------------------------------------
# Two runs sharing an OUTPUT_DIR is silent, total data loss.  Both write
# model.pt and model.pt.ep{N} to the same paths, and both run funasr's
# keep_nbest pruning, whose Trainer.save_checkpoint os.remove()s the checkpoints
# that lost the ranking -- including ones the other process is midway through
# writing.  Nothing in either log says anything is wrong; the damage only shows
# up as a corrupt torch.load hours later, after the compute is gone.
#
# Mechanism: an atomic mkdir plus a pid file inside it.  macOS ships neither
# flock(1) (util-linux) nor lockfile(1) (procmail).  /usr/bin/shlock does exist
# and does its own stale-pid check, but it is deprecated on this platform, is
# absent everywhere that is not macOS/BSD, and has no release side at all -- the
# caller still deletes the file by hand, so it would not spare us the ownership
# check below.  mkdir(2) is atomic on every POSIX filesystem and needs nothing
# installed, so the lock is portable and the semantics are ours.
LOCK_DIR=""
LOCK_PID_FILE=""
LOCK_HELD=0

# Only ever release a lock THIS process owns.  If the pid file no longer names
# us, some other run took the directory over (we were SIGKILLed and looked
# stale, say) and removing it would hand a third run a lock the second one is
# actively using -- the precise failure this whole section exists to prevent.
lock_release() {
    local rc=$? owner=""
    if [ "${LOCK_HELD}" = "1" ]; then
        LOCK_HELD=0
        if [ -s "${LOCK_PID_FILE}" ]; then
            read -r owner <"${LOCK_PID_FILE}" 2>/dev/null || owner=""
        fi
        if [ "${owner}" = "$$" ]; then
            rm -f "${LOCK_PID_FILE}"
            rmdir "${LOCK_DIR}" 2>/dev/null || true
        else
            warn "lock ${LOCK_DIR} is now held by pid '${owner}', not by us ($$); leaving it alone"
        fi
    fi
    return "${rc}"
}

# Does this pid exist?  NOT `kill -0` alone: that returns failure both for "no
# such process" (ESRCH) and for "process exists but you do not own it" (EPERM),
# and treating the second as a dead holder would make us steal a lock that is
# very much in use.  ps -p reports any process regardless of ownership; kill -0
# stays as the fallback for the case where ps is unavailable or restricted.
pid_is_alive() {
    local pid="$1"
    [ -n "${pid}" ] || return 1
    ps -p "${pid}" -o pid= >/dev/null 2>&1 && return 0
    kill -0 "${pid}" 2>/dev/null
}

# First line of the pid file, or empty.  Retried: a competing run can be between
# its mkdir and its pid-file write, and treating that half-built lock as stale
# would let both processes think they hold it.
lock_read_pid() {
    local pid="" i
    for i in 1 2 3 4 5; do
        if [ -s "${LOCK_PID_FILE}" ]; then
            read -r pid <"${LOCK_PID_FILE}" 2>/dev/null || pid=""
            case "${pid}" in
                ''|*[!0-9]*) pid="" ;;
                *) printf '%s' "${pid}"; return 0 ;;
            esac
        fi
        if [ "${i}" -lt 5 ]; then
            sleep 0.2
        fi
    done
    return 1
}

# Claim a stale lock by renaming it first.  rename(2) has exactly one winner, so
# if two runs spot the same dead lock simultaneously only one gets to clear it;
# the loser's mv fails, it loops, and it then contends on the normal mkdir.
# Without this both would rmdir-then-mkdir and both would believe they won.
lock_break() {
    local grave="${LOCK_DIR}.stale.$$.$(date +%s)"
    mv "${LOCK_DIR}" "${grave}" 2>/dev/null || return 1
    rm -f "${grave}/pid"
    # rmdir, not rm -rf: if anything unexpected is in there, fail loudly and
    # leave it for a human rather than deleting it.
    rmdir "${grave}" 2>/dev/null \
        || warn "stale lock remains at ${grave} (unexpected contents); remove it by hand"
    return 0
}

lock_acquire() {
    local dir="$1" attempt holder holder_cmd
    LOCK_DIR="${dir}/.train.lock"
    LOCK_PID_FILE="${LOCK_DIR}/pid"

    # Installed before the first mkdir and a no-op until LOCK_HELD flips, so the
    # lock cannot outlive us via a signal delivered just after we take it.
    trap 'lock_release' EXIT
    trap 'lock_release; trap - INT; kill -INT $$' INT
    trap 'lock_release; trap - TERM; kill -TERM $$' TERM

    for attempt in 1 2 3; do
        if mkdir "${LOCK_DIR}" 2>/dev/null; then
            LOCK_HELD=1
            printf '%s\n' "$$" >"${LOCK_PID_FILE}"
            printf 'host=%s\nstarted=%s\nscript=%s\n' \
                "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BASH_SOURCE[0]}" \
                >>"${LOCK_PID_FILE}"
            info "lock:       ${LOCK_DIR} (pid $$)"
            return 0
        fi

        holder="$(lock_read_pid)" || holder=""

        if pid_is_alive "${holder}"; then
            holder_cmd="$(ps -p "${holder}" -o command= 2>/dev/null || true)"
            die "$(printf '%s\n' \
                "another run already holds ${LOCK_DIR}" \
                "" \
                "  holder pid:  ${holder}${holder_cmd:+  (${holder_cmd})}" \
                "  output_dir:  ${dir}" \
                "" \
                "Two runs sharing an output directory overwrite each other's model.pt and" \
                "delete each other's per-epoch checkpoints through funasr's keep_nbest" \
                "pruning, which destroys both runs silently.  Refusing to start." \
                "" \
                "Either wait for pid ${holder} to finish, or point this run somewhere else:" \
                "  OUTPUT_DIR=${dir}_2 ${BASH_SOURCE[0]}" \
                "" \
                "If pid ${holder} is definitely not a training run, delete ${LOCK_DIR}.")"
        fi

        if [ -z "${holder}" ]; then
            warn "lock ${LOCK_DIR} has no readable pid file; treating it as abandoned"
        else
            warn "lock ${LOCK_DIR} was held by pid ${holder}, which is no longer running (crashed run); taking it over"
        fi
        lock_break || true
    done

    die "could not acquire ${LOCK_DIR} after 3 attempts; another run is contending for it"
}

# Read-only view of the lock, for --dry-run.  Reports without creating anything.
lock_status() {
    local dir="$1" holder
    LOCK_DIR="${dir}/.train.lock"
    LOCK_PID_FILE="${LOCK_DIR}/pid"
    [ -d "${LOCK_DIR}" ] || return 0
    holder="$(lock_read_pid)" || holder=""
    if pid_is_alive "${holder}"; then
        warn "${dir} is currently locked by a live run (pid ${holder}); a real run would refuse to start until it finishes"
    else
        info "  note: a stale lock is present at ${LOCK_DIR}; a real run would take it over"
    fi
}

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
print_command() {
    local i
    for assignment in "${ENV_ASSIGNMENTS[@]}"; do
        printf 'export %s\n' "${assignment}"
    done
    for (( i = 0; i < ${#CMD[@]}; i++ )); do
        if [ "${i}" -eq 0 ]; then
            printf '%s' "$(printf '%q' "${CMD[$i]}")"
        else
            printf ' \\\n    %s' "$(printf '%q' "${CMD[$i]}")"
        fi
    done
    printf ' \\\n    2>&1 | tee %s\n' "$(printf '%q' "${log_file}")"
}

if [ "${DRY_RUN}" = "1" ]; then
    info ""
    # Reports the lock, never takes one: --dry-run must not be able to lock out
    # the real run it is being used to preview.
    lock_status "${OUTPUT_DIR}"
    info "resolved command (--dry-run, nothing was executed):"
    info ""
    print_command
    info ""
    if [ "${soft_failures}" -gt 0 ]; then
        info "dry run finished with ${soft_failures} check(s) that WOULD ABORT a real run."
        exit 0
    fi
    info "dry run finished: all checks passed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# Existing runs under OUTPUT_DIR are left alone; only a new log file is added,
# which is what makes train_conf.resume=true usable across invocations.  That
# tolerance is exactly why the lock is needed: resume=true means a second run
# aimed here does not announce itself, it silently adopts and then corrupts the
# first run's checkpoints.  Take the lock before creating anything else, and
# hold it for the lifetime of the training process.
mkdir -p "${OUTPUT_DIR}"
lock_acquire "${OUTPUT_DIR}"

# Under the lock, so it cannot interleave with another run's record.
write_chunk_geometry_json "${OUTPUT_DIR}/chunk_geometry.json"

mkdir -p "${hydra_run_dir}"

info ""
info "output_dir: ${OUTPUT_DIR}"
info "geometry:   ${OUTPUT_DIR}/chunk_geometry.json"
info "log_file:   ${log_file}"
info "launching:"
print_command
info ""

"${CMD[@]}" 2>&1 | tee "${log_file}"
