#!/usr/bin/env bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
#
# Dynamic chunk-mask finetuning for SenseVoiceSmall.
#
# Derived from finetune.sh, which stays untouched.  Two differences matter:
#   1. the encoder is configured for dynamic chunk masking (see CHUNK_ARGS below)
#   2. SMOKE=1 runs a CPU-only, few-step pipeline check on locally generated data
#
# Usage:
#   ./finetune_chunk.sh              # normal GPU training
#   SMOKE=1 ./finetune_chunk.sh      # CPU smoke run (plumbing check only)

set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Dynamic chunk masking
# ---------------------------------------------------------------------------
# These map onto the ``encoder_conf`` fields added in model.py.  They are kept
# here as a single block because the exact parameter names are still settling;
# if model.py renames a field, edit only this array.
#
# Hydra parses ``[a,b,c]`` as a list, but bare brackets are a glob pattern to the
# shell, so each list value must stay quoted all the way to the trainer -- hence
# the quoted array elements plus "${CHUNK_ARGS[@]}" at the call site.
CHUNK_ARGS=(
    "++encoder_conf.chunk_size=[8,12,16]"
    "++encoder_conf.stride=[6,10,14]"
    "++encoder_conf.pad_left=[0,0,0]"
    "++encoder_conf.encoder_att_look_back_factor=[1,1,1]"
)

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${workspace}/.venv/bin/python" ]; then
        PYTHON_BIN="${workspace}/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi
echo "python: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# Model: prefer a local directory over a ModelScope/HF download
# ---------------------------------------------------------------------------
# README.md ("Using a local model directory") explains why: funasr pip-installs
# the requirements.txt bundled inside a downloaded model directory, which can
# override this repository's dependency pins.  models/ is gitignored.
local_model_dir="${workspace}/models/SenseVoiceSmall"
if [ -n "${MODEL_DIR:-}" ]; then
    model_name_or_model_dir="${MODEL_DIR}"
elif [ -d "${local_model_dir}" ]; then
    model_name_or_model_dir="${local_model_dir}"
else
    model_name_or_model_dir="iic/SenseVoiceSmall"
fi
echo "model: ${model_name_or_model_dir}"

# ---------------------------------------------------------------------------
# funasr trainer entry point
# ---------------------------------------------------------------------------
find_train_tool() {
    local funasr_bin funasr_pkg candidate
    if funasr_bin="$(command -v funasr 2>/dev/null)"; then
        candidate="$(dirname "${funasr_bin}")/train_ds.py"
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    # Fall back to the installed package location.
    if funasr_pkg="$("${PYTHON_BIN}" -c 'import funasr, os; print(os.path.dirname(funasr.__file__))' 2>/dev/null)"; then
        candidate="${funasr_pkg}/bin/train_ds.py"
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi
    return 1
}

if ! train_tool="$(find_train_tool)"; then
    echo "Error: train_ds.py not found. Is funasr installed in ${PYTHON_BIN}?" >&2
    exit 1
fi
echo "trainer: ${train_tool}"

# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
    # CPU pipeline check: confirms the chunk-mask config is accepted and that a
    # step runs end to end.  It says nothing about model quality.
    echo "mode: SMOKE (CPU, few steps)"
    export CUDA_VISIBLE_DEVICES=""
    nproc_per_node=1

    train_data="${workspace}/data/smoke_train.jsonl"
    val_data="${workspace}/data/smoke_val.jsonl"
    echo "generating smoke data..."
    "${PYTHON_BIN}" "${workspace}/scripts/make_smoke_data.py" \
        --train-jsonl "${train_data}" \
        --val-jsonl "${val_data}"

    output_dir="${workspace}/outputs/chunk_smoke"
    # Small example-batches with a small sort window, so a batch still mixes
    # clip lengths and the padding path gets exercised.
    DATA_ARGS=(
        "++dataset_conf.batch_type=example"
        "++dataset_conf.batch_size=4"
        "++dataset_conf.sort_size=4"
        "++dataset_conf.num_workers=0"
    )
    TRAIN_ARGS=(
        "++train_conf.max_epoch=1"
        "++train_conf.log_interval=1"
        "++train_conf.resume=false"
        "++train_conf.validate_interval=2"
        "++train_conf.save_checkpoint_interval=2"
        "++train_conf.keep_nbest_models=1"
        "++train_conf.avg_nbest_model=1"
        "++train_conf.use_deepspeed=false"
    )
else
    echo "mode: TRAIN (GPU)"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    nproc_per_node="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')"

    train_data="${workspace}/data/train_example.jsonl"
    val_data="${workspace}/data/val_example.jsonl"
    output_dir="${workspace}/outputs/chunk"
    deepspeed_config="${workspace}/deepspeed_conf/ds_stage1.json"
    DATA_ARGS=(
        "++dataset_conf.batch_type=token"
        "++dataset_conf.batch_size=6000"
        "++dataset_conf.sort_size=1024"
        "++dataset_conf.num_workers=4"
    )
    TRAIN_ARGS=(
        "++train_conf.max_epoch=50"
        "++train_conf.log_interval=1"
        "++train_conf.resume=true"
        "++train_conf.validate_interval=2000"
        "++train_conf.save_checkpoint_interval=2000"
        "++train_conf.keep_nbest_models=20"
        "++train_conf.avg_nbest_model=10"
        "++train_conf.use_deepspeed=false"
        "++train_conf.deepspeed_config=${deepspeed_config}"
    )
fi

for data_file in "${train_data}" "${val_data}"; do
    if [ ! -f "${data_file}" ]; then
        echo "Error: data file not found: ${data_file}" >&2
        exit 1
    fi
done

# Existing runs under output_dir are left alone; only a new log file is added.
mkdir -p "${output_dir}"
log_file="${output_dir}/log_$(date +%Y%m%d_%H%M%S).txt"
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS=(
    --nnodes "${WORLD_SIZE:-1}"
    --nproc_per_node "${nproc_per_node}"
    --node_rank "${RANK:-0}"
    --master_addr "${MASTER_ADDR:-127.0.0.1}"
    --master_port "${MASTER_PORT:-26669}"
)

# torchrun via -m keeps the run on ${PYTHON_BIN} even when the venv is not active.
"${PYTHON_BIN}" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "${train_tool}" \
    ++model="${model_name_or_model_dir}" \
    ++trust_remote_code=true \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset_conf.data_split_num=1 \
    ++dataset_conf.batch_sampler="BatchSampler" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHUNK_ARGS[@]}" \
    ++optim_conf.lr=0.0002 \
    ++output_dir="${output_dir}" 2>&1 | tee "${log_file}"
