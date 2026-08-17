#!/usr/bin/env bash
#SBATCH --job-name=sv-chunk
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# SITE-SPECIFIC: THE FOUR LINES BELOW MUST BE EDITED BEFORE THE FIRST RUN
# ---------------------------------------------------------------------------
# The image reference and the three bind mounts that follow carry placeholders
# and are the only thing in this repo that is not portable:
#
#   <cluster-registry>  the container registry host the image is pulled from
#   <project>           the registry namespace/project holding that image
#   <user>              the account directory on the shared filesystem
#
# Replace all three with your site's values.  Do NOT change the image name and
# tag (sensevoice-chunk:24.12-funasr1.4.1-v1) or the container-side mount targets
# (/workspace, /corpus, /outputs): the CONFIGURATION defaults, the preflight and
# the docs all refer to those targets, and /corpus in particular is a deliberate
# choice explained immediately below.
#
# WHY THESE FOUR CANNOT BE ENVIRONMENT VARIABLES
# Every other setting in this script is an environment variable with a default.
# These four are the sole exception, and not by oversight.  Submission goes
# through the site's `sbatch` wrapper, which reads the "#SBATCH" and "#CONTAINER"
# lines out of this file as TEXT -- it docker-pulls the --container image and
# assembles the bind mounts before bash is ever started.  So nothing on those
# lines can be templated: "${VAR}" written there is eight literal characters, not
# a substitution, and there is no later expansion step that would ever turn it
# into one.  Editing the file is the only channel that exists.  To keep a site
# edit out of commits: `git update-index --skip-worktree finetune_chunk_slurm.sh`,
# or carry the edit on a site branch.
#
# The check further down reads these lines back out of this file and refuses to
# start while a placeholder is still present, so an unedited checkout fails in
# the first second with a message naming the lines -- rather than after a queue
# wait, or with the workspace resolver's advice to export WORKSPACE, which is the
# wrong problem entirely when the mounts were never applied in the first place.
# ---------------------------------------------------------------------------
#SBATCH --container=<cluster-registry>/<project>/sensevoice-chunk:24.12-funasr1.4.1-v1
#CONTAINER -v /home/share/<user>/sensevoice/SenseVoice-mod:/workspace
# The corpus is mounted at /corpus, NOT the obvious /data: MEASURED on a compute
# node, the site mounts a local scratch filesystem (/dev/md0, ext4) at /data
# AFTER our bind mounts are applied, which shadows anything bound there.  The
# failure is silent -- /data exists, is readable, and lists someone else's
# datasets (MNIST, midas_data, share_data), so the only symptom is the corpus
# "disappearing" and preflight aborting on a missing manifest after the job has
# already waited in the queue.  /workspace and /outputs are not affected.
#CONTAINER -v /home/share/<user>/data:/corpus
#CONTAINER -v /home/share/<user>/outputs:/outputs
#CONTAINER --shm-size 32G
#CONTAINER --disable-entrypoint
#
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
#
# Dynamic chunk-mask finetuning for SenseVoiceSmall on the university H100 cluster.
#
# This is the Slurm sibling of finetune_chunk_mps.sh.  It follows the GPU path of
# finetune_chunk.sh -- torchrun onto funasr's bin/train_ds.py -- because that is
# the trainer that actually works on CUDA: train_ds.py sets
#     kwargs["device"] = int(os.environ.get("LOCAL_RANK", 0))
#     trainer.device   = int(os.environ.get("LOCAL_RANK", 0))
# right after warp_model, which is correct here (one rank per GPU) and is exactly
# what made it unusable on MPS.  Everything else -- model, dataset, sampler, chunk
# geometry, LR ceiling, OUTPUT_DIR lock, preflight -- is shared with the MPS
# script so the two runs stay comparable and the two files read as siblings.
#
# ---------------------------------------------------------------------------
# WHY THE SBATCH HEADER LOOKS LIKE THIS
# ---------------------------------------------------------------------------
# PROVENANCE OF THE NUMBERS BELOW -- READ THIS BEFORE QUOTING THEM
# An earlier version of this block asserted a set of `research` QOS limits with
# no indication of where they came from or whether anyone had ever checked.  They
# were wrong, and because they read as fact they were quoted back as fact.  So:
# every claim here is now tagged MEASURED or INFERRED, and the commands and date
# are recorded so the next reader can re-run them rather than trust this comment.
#
# MEASURED on 2026-08-18 with:
#     sacctmgr show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU%40,MaxTRESPerJob%40
#     scontrol show partition research
#     sacct --starttime 2026-08-01 -o JobID,JobName,QOS,Partition,Timelimit,Elapsed,State
#
# * THIS ACCOUNT RUNS UNDER QOS=normal, NOT `research`.  The partition advertises
#   a research QoS, but the association resolves to `normal`, and the evidence
#   that settles it is the scheduler's own record rather than any configuration:
#   the two completed training jobs, 12899 and 12904, are logged by sacct with
#   QOS=normal.  So the `research` QOS limits this comment used to assert (4 GPUs
#   per user, 16 CPUs, MaxSubmitJobsPU=4) are most likely NOT the binding
#   constraint for this account at all.  If you are adapting this script for a
#   different account, re-run the commands above before assuming anything here
#   applies to you.
# * WHAT ACTUALLY BINDS IS NODE CAPACITY (MEASURED): partition `research` has one
#   node, `t-gpu01`, with 8 GPUs and 32 CPUs, shared with other users.
# * CONCURRENCY IS A CONVENTION, NOT AN ENFORCED LIMIT (INFERRED).  No per-user
#   job cap has been confirmed for QOS=normal, so do not treat any number as
#   enforced -- what governs how much runs at once is node capacity and
#   contention with other users.  The project's operating convention is a
#   conservative 2-3 jobs, which fits inside the node: training on 2 GPUs,
#   labelling on 1, evaluation on 1.
# * --cpus-per-task=12 leaves room on the 32-CPU node for a labelling or
#   evaluation job to run alongside this one.  It is sized to the convention
#   above, not to a per-user CPU quota (the 16-CPU quota it used to cite has not
#   been confirmed to apply).
# * --gpus-per-task=2 with --nodes=1 --ntasks=1 is the GRES form verified working
#   at this site.  --gres=gpu:N is NOT what the working site example used; do not
#   "simplify" it back to that without re-testing.
# * THE WALL CLOCK IS THE DANGEROUS ONE (MEASURED).  This header requests
#   --time=24:00:00, but sacct records jobs 12899 and 12904 with
#   Timelimit=08:00:00 -- i.e. the request is being CLAMPED to 8 h and the 24 h
#   this file's documentation long assumed was never granted.  Round 2 finished
#   in about 2 h so it never mattered; round 3 trains on a larger corpus, where
#   it does.  A run truncated at the ceiling looks like a COMPLETED job with a
#   missing final epoch, and this site's scheduler discards the batch script's
#   exit code, so nothing in sacct distinguishes it from success.  The preflight
#   now compares the requested --time against TIME_CEILING_HOURS and says so.
#   THE MITIGATION IS THE RESUME CHAIN, and it is not optional for a long round:
#   scripts/submit_chunk_chain.sh submits N links with --dependency=afterany, and
#   each link resumes from the last checkpoint (RESUME=true reading
#   ${OUTPUT_DIR}/model.pt), so the run survives being cut at the ceiling.
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT NEVER READS SLURM_JOB_ID
# ---------------------------------------------------------------------------
# Submission goes through the site's custom `sbatch` wrapper (a Python script
# that docker-pulls the --container image, applies the #CONTAINER options and
# forwards the rest to the real sbatch).  MEASURED on this cluster: inside the
# container SLURM_JOB_ID and every other SLURM_* variable is UNSET -- the wrapper
# does not forward the job environment.  CUDA_VISIBLE_DEVICES *is* set correctly.
# Therefore:
#   * the GPU count is derived from CUDA_VISIBLE_DEVICES (then `nvidia-smi -L`,
#     then 1), never from SLURM_GPUS_* or SLURM_NTASKS;
#   * every SLURM_* read is guarded with ${VAR:-} and is for logging only;
#   * the OUTPUT_DIR lock identifies its holder by hostname:pid plus a heartbeat
#     file, because a job id is not available and the holder is usually on a
#     different host than whoever is inspecting the lock.
#
# Usage (inside the container, or as an sbatch script):
#   sbatch finetune_chunk_slurm.sh          # normal 2-GPU run
#   ./finetune_chunk_slurm.sh --dry-run     # preflight + print resolved command
#   SMOKE=1 sbatch finetune_chunk_slurm.sh  # minutes-long container smoke test
#   MAX_EPOCH=1 sbatch finetune_chunk_slurm.sh   # override any setting below
#
# Every setting is an environment variable with a default; see CONFIGURATION.
# The four site-specific header lines at the top of this file are the sole
# exception -- they must be edited in place before the first run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0

usage() {
    cat <<'EOF'
Dynamic chunk-mask finetuning for SenseVoiceSmall on the H100 Slurm cluster.

Usage:
  sbatch finetune_chunk_slurm.sh          # normal 2-GPU run
  ./finetune_chunk_slurm.sh --dry-run     # preflight + print resolved command
  SMOKE=1 sbatch finetune_chunk_slurm.sh  # minutes-long container smoke test

Options:
  --dry-run    Run every preflight check, report failures as warnings instead
               of aborting, print the fully resolved command, and exit without
               taking the lock or launching training.
  -h, --help   This message.

Site-specific setup (required once, before the first run):
  The --container= image line and the three bind-mount lines at the top of this
  file contain <cluster-registry>, <project> and <user> placeholders and must be
  edited to your site's values.  They are the only settings here that cannot come
  from the environment: the site sbatch wrapper parses those lines as text and
  pulls the image before bash starts, so nothing in them can be templated.  The
  run refuses to start while a placeholder is still present.

Environment overrides (defaults in the CONFIGURATION block of this file):
  WORKSPACE (repo root; required under sbatch if the repo is not at /workspace)
  MODEL_DIR TRAIN_JSONL VAL_JSONL OUTPUT_DIR DEVICE MAX_EPOCH LR BATCH_TOKENS
  NUM_WORKERS SAVE_INTERVAL KEEP_NBEST RESUME USE_BF16 SMOKE NPROC_PER_NODE
  CHUNK_SIZES STRIDES PAD_LEFTS LOOK_BACKS PYTHON_BIN LR_CEILING
  EXPECT_TRAIN_HOURS EXPECT_TRAIN_HOURS_TOLERANCE EXPECT_VAL_CLIPS
  INIT_PARAM RICH_WEIGHT EMO_MASK_TOKEN_ID TIME_CEILING_HOURS

  TIME_CEILING_HOURS (default 8) is the wall clock this script believes the
  scheduler will actually grant.  The preflight compares it against the
  '#SBATCH --time=' this file requests and warns when the request is larger,
  because a job cut off at the ceiling leaves a COMPLETED record with a missing
  final epoch and this site's scheduler discards exit codes, so nothing
  downstream can tell it from a clean finish.  The default records an
  observation, not an enforced limit -- sacct logs jobs 12899/12904 with
  Timelimit=08:00:00 against a 24 h request -- and the enforced value cannot be
  read from inside the container, since the site wrapper forwards no SLURM_*.
  Set it to whatever you have confirmed and the warning goes quiet.  The
  mitigation for a long round is scripts/submit_chunk_chain.sh, which chains
  jobs with --dependency=afterany so each link resumes from the last checkpoint.

  INIT_PARAM, RICH_WEIGHT and EMO_MASK_TOKEN_ID are all unset by default and add
  nothing to the command when unset, so an invocation that sets none of them runs
  exactly the command earlier rounds ran.

  INIT_PARAM is a checkpoint to initialise the weights from (++init_param), i.e.
  how round N starts from a chosen epoch of round N-1.  The preflight refuses a
  path that does not exist, because funasr does not: it prints one line and
  trains from scratch, costing the whole allocation.  RESUME defaults to true and
  the two coexist -- init_param is applied at model build and a resumed
  ${OUTPUT_DIR}/model.pt overwrites it afterwards, so a fresh OUTPUT_DIR seeds
  from INIT_PARAM while a restarted job keeps its own progress.  The preflight
  says which one supplies the weights; it never fails on the combination.

  RICH_WEIGHT sets ++model_conf.rich_loss_weight (non-negative float).
  EMO_MASK_TOKEN_ID sets ++model_conf.emo_mask_token_id (non-negative integer);
  24991 is the id of <|SER|> and maps the emotion slot to ignore_id so it drops
  out of the loss.  SMOKE keeps both -- proving they load is the point of a smoke
  run for this change -- but drops INIT_PARAM.

  The preflight also reports a manifest whose emo_target never varies: that is the
  round-1/2 defect, where every clip was <|NEUTRAL|>, the emotion head trained on
  a constant and acc_rich saturated at 1.0 while nothing had been learned.  It is
  a note normally, and a hard failure when EMO_MASK_TOKEN_ID is set outside SMOKE,
  since masking against constant labels is a contradiction.

  It also cross-checks the <|SER|> sentinel against EMO_MASK_TOKEN_ID, because the
  two are chosen in different places (the manifest by prepare_vn_data.py
  --emo-labels, the mask here) and nothing else connects them.  A manifest holding
  <|SER|> with EMO_MASK_TOKEN_ID unset is FATAL in every mode including SMOKE:
  model.py raises on it at training step 1 anyway, so the only question is whether
  that happens before or after the GPU allocation is spent.  The converse --
  EMO_MASK_TOKEN_ID set against a manifest with no <|SER|> at all -- is a warning,
  since it is usually the round-2 constant-<|NEUTRAL|> corpus but is legitimate
  for an ablation.  Both counts are exact, not sampled.

  EXPECT_TRAIN_HOURS is unset by default and opt-in: set it to the audio-hours
  the training manifest is supposed to hold and the preflight refuses to start
  when the manifest disagrees by more than EXPECT_TRAIN_HOURS_TOLERANCE (a
  fraction, default 0.05 = +/-5%).  EXPECT_VAL_CLIPS is the exact clip count the
  validation manifest must have -- exact, because val is pinned so that
  round-to-round numbers stay comparable.  Both exist because a run on a manifest
  that was silently rebuilt over a subset of the archives SUCCEEDS: it trains,
  exits 0 and reports preflight=passed, so no other check and no job-status
  sentinel can distinguish it from the intended run.  Leave them unset and the
  preflight behaves exactly as it did before.

  LR_CEILING defaults to 0.0002 and the run refuses to start when LR exceeds it.
  Raising it is allowed but deliberate: pass LR_CEILING explicitly (e.g.
  LR=0.0006 LR_CEILING=0.0008) and the run warns loudly and records the ceiling
  in ${OUTPUT_DIR}/chunk_geometry.json.
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
# Job outcome: status sentinel + terminal log marker
# ---------------------------------------------------------------------------
# MEASURED on this cluster (job 12878): the batch script's exit code is
# DISCARDED.  A job whose entire body was `echo ...; exit 42` logged its echo and
# was then recorded as
#     12878        COMPLETED   ExitCode 0:0   DerivedExitCode 0:0
#     12878.batch  COMPLETED   ExitCode 0:0
# The site's OCI/container integration masks the script's status, so EVERY job
# reports COMPLETED regardless of what happened inside.  That is also why the
# smoke job that died on the missing make_smoke_data.py (12876) was recorded as
# COMPLETED 0:0.  sacct/squeue state is therefore worthless as a success signal
# here: a 24 h run that crashed in hour 2 is indistinguishable from one that
# finished, to a human and to any monitoring built on job state.
#
# So the job reports its own outcome through two channels the scheduler cannot
# mask:
#   1. ${OUTPUT_DIR}/.job_status -- key=value lines, written atomically so a
#      reader racing the exit sees the previous run's record or this one's,
#      never a half-written file.
#   2. a terminal marker line, the last thing the job prints, so that tailing
#      or grepping the log is a reliable monitor.  Before this, a failed job
#      simply stopped mid-log with no terminal line at all.
# Both are emitted from the EXIT trap, so they cover `set -e` aborts, crashes
# and signals -- not just the happy path.
JOB_MARKER_SUCCESS="SENSEVOICE_JOB_OK"
JOB_MARKER_FAILURE="SENSEVOICE_JOB_FAILED"
JOB_STATUS_BASENAME=".job_status"
JOB_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# passed | failed | not_reached.  Recorded so a reader can tell "the checks said
# no" from "it died before ever reaching the checks".
PREFLIGHT_STATUS="not_reached"

job_mode() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN'
    elif [ "${SMOKE:-0}" = "1" ]; then
        printf 'SMOKE'
    else
        printf 'TRAIN'
    fi
}

# Writes ${OUTPUT_DIR}/.job_status.  Returns non-zero (quietly) if there is
# nowhere to write it yet -- an early abort can happen before OUTPUT_DIR is even
# resolved, and the marker line still reports that case.
job_status_write() {
    local rc="$1" reason="$2" result="$3" mode="$4" dest tmp marker
    if [ "${DRY_RUN}" = "1" ]; then
        return 1
    fi
    if [ "${result}" = "SUCCESS" ]; then
        marker="${JOB_MARKER_SUCCESS}"
    else
        marker="${JOB_MARKER_FAILURE}"
    fi
    if [ -z "${OUTPUT_DIR:-}" ]; then
        return 1
    fi
    dest="${OUTPUT_DIR}/${JOB_STATUS_BASENAME}"
    tmp="${dest}.tmp.$$"
    mkdir -p "${OUTPUT_DIR}" 2>/dev/null || return 1
    {
        printf 'schema_version=1\n'
        printf 'written_by=finetune_chunk_slurm.sh\n'
        printf 'result=%s\n' "${result}"
        printf 'exit_code=%s\n' "${rc}"
        printf 'reason=%s\n' "${reason}"
        printf 'mode=%s\n' "${mode}"
        printf 'preflight=%s\n' "${PREFLIGHT_STATUS}"
        printf 'started_at=%s\n' "${JOB_STARTED_AT}"
        printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'host=%s\n' "$(hostname)"
        printf 'pid=%s\n' "$$"
        printf 'run_id=%s\n' "${RUN_ID:-}"
        printf 'output_dir=%s\n' "${OUTPUT_DIR}"
        printf 'workspace=%s\n' "${workspace:-}"
        printf 'log_file=%s\n' "${log_file:-}"
        printf 'marker=%s\n' "${marker}"
    } >"${tmp}" 2>/dev/null || { rm -f "${tmp}" 2>/dev/null || true; return 1; }
    mv "${tmp}" "${dest}" 2>/dev/null || { rm -f "${tmp}" 2>/dev/null || true; return 1; }
    return 0
}

job_finish() {
    local rc="$1" reason="$2" result marker mode dest=""
    if [ "${rc}" = "0" ]; then
        result="SUCCESS"
        marker="${JOB_MARKER_SUCCESS}"
    else
        result="FAILURE"
        marker="${JOB_MARKER_FAILURE}"
    fi
    mode="$(job_mode)"
    if job_status_write "${rc}" "${reason}" "${result}" "${mode}"; then
        dest="${OUTPUT_DIR}/${JOB_STATUS_BASENAME}"
    fi
    # Single line, fixed leading token, no leading whitespace: greppable with a
    # plain fixed-string match from outside.
    printf '%s rc=%s mode=%s preflight=%s reason=%s host=%s run_id=%s output_dir=%s status_file=%s finished_at=%s\n' \
        "${marker}" "${rc}" "${mode}" "${PREFLIGHT_STATUS}" "${reason}" \
        "$(hostname)" "${RUN_ID:-<unset>}" "${OUTPUT_DIR:-<unresolved>}" \
        "${dest:-<none>}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# rc is captured on the FIRST line, before any cleanup command can overwrite $?.
# Getting this wrong is exactly how a status sentinel ends up reporting 0 for
# every run -- i.e. how it reproduces the scheduler bug it exists to work around.
on_exit() {
    local rc=$?
    trap - EXIT
    # The lock section is defined much further down; an abort before that point
    # has no lock to release.
    if declare -F lock_release >/dev/null 2>&1; then
        lock_release || true
    fi
    job_finish "${rc}" "exit" || true
    exit "${rc}"
}

# Signals get 128+n, the shell's own convention, so that the wall-clock SIGTERM
# Slurm sends at 24 h stays distinguishable from an ordinary crash.  The signal
# is re-raised afterwards so the process still dies of what killed it.
on_signal() {
    local sig="$1" num="$2"
    local rc=$(( 128 + num ))
    trap - EXIT
    if declare -F lock_release >/dev/null 2>&1; then
        lock_release || true
    fi
    job_finish "${rc}" "signal:SIG${sig}" || true
    trap - "${sig}"
    kill -"${sig}" "$$"
}

# Installed here, before anything that can fail, so that even an abort in the
# workspace resolution below produces a terminal marker.  --dry-run reaches this
# too: it prints the marker (mode=DRY_RUN) but job_status_write refuses to write
# anything, because --dry-run must not touch the filesystem.
trap 'on_exit' EXIT
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM

# ---------------------------------------------------------------------------
# Site-specific header lines: refuse to run on an unedited checkout
# ---------------------------------------------------------------------------
# See the SITE-SPECIFIC block at the top of this file: the --container= image and
# the three bind mounts are the only settings that cannot be supplied from the
# environment, because the site sbatch wrapper reads them as text before bash
# runs.  A checkout that still has the placeholders therefore CANNOT work, and
# every symptom it produces points somewhere else:
#   * a real submission never reaches this script at all -- the wrapper fails
#     trying to pull an image whose host is literally "<cluster-registry>";
#   * anything that does start has no mounts, so the workspace resolver below
#     aborts with a long message advising `export WORKSPACE=/workspace`, which is
#     useless when the bind mount was never applied;
#   * with WORKSPACE set by hand it survives further still and dies on a missing
#     MODEL_DIR or a missing corpus manifest, i.e. three layers from the cause.
# So this runs FIRST, ahead of everything that could mis-diagnose it, and reports
# through the same check_fail path as every other check: a real run exits here
# and the EXIT trap records the refusal in the outcome sentinel like any other,
# while --dry-run downgrades it to a warning so the resolved command can still be
# inspected from a laptop checkout (where the placeholders are expected).
#
# The file inspected is ${BASH_SOURCE[0]}.  Under sbatch that is the scheduler's
# spool copy rather than the repo file -- which is the correct choice, not merely
# a convenient one: the spool copy is the exact text the wrapper parsed.
SITE_HEADER_PATTERN='^#(SBATCH --container=|CONTAINER -v )'

site_header_check() {
    local src="${BASH_SOURCE[0]}" line found=0 bad=0 listing=""

    if [ ! -r "${src}" ]; then
        warn "cannot read ${src} to verify the site-specific header lines; check the --container= and bind-mount lines at the top of finetune_chunk_slurm.sh by hand"
        return 0
    fi

    # `|| true` because grep exits 1 on no match, which is a state this function
    # reports rather than an error that should trip set -e.
    while IFS= read -r line; do
        found=$(( found + 1 ))
        case "${line}" in
            # A literal '<' cannot occur in a real registry host, image
            # reference or absolute path, so it is an unambiguous marker of an
            # unedited placeholder.
            *'<'*)
                bad=$(( bad + 1 ))
                listing="${listing}    ${line}"$'\n'
                ;;
        esac
    done < <(grep -E "${SITE_HEADER_PATTERN}" "${src}" 2>/dev/null || true)

    if [ "${found}" -eq 0 ]; then
        warn "no --container= or bind-mount header lines found in ${src}; the site sbatch wrapper would launch this job with no repo, corpus or output mount at all"
        return 0
    fi

    [ "${bad}" -gt 0 ] || return 0

    check_fail "$(printf '%s\n' \
        "the site-specific header lines in this file are still placeholders. Edit them before submitting." \
        "" \
        "  file: ${src}" \
        "  unedited line(s):" \
        "${listing%$'\n'}" \
        "" \
        "Replace <cluster-registry> with your container registry host, <project> with" \
        "the registry namespace holding the image, and <user> with your account" \
        "directory on the shared filesystem.  Keep the image name and tag" \
        "(sensevoice-chunk:24.12-funasr1.4.1-v1) and the container-side mount targets" \
        "(/workspace, /corpus, /outputs) exactly as they are -- the CONFIGURATION" \
        "defaults, the preflight and the docs all refer to those targets." \
        "" \
        "These are the ONLY settings in this script that cannot come from the" \
        "environment: the site sbatch wrapper parses them as text and pulls the image" \
        "before bash starts, so a variable reference written there stays four literal" \
        "characters.  Every other setting is an environment variable with a default" \
        "(see the CONFIGURATION block, or --help)." \
        "" \
        "Full explanation: the SITE-SPECIFIC block at the top of this file.")"
}

site_header_check

# ---------------------------------------------------------------------------
# Workspace (the repo root)
# ---------------------------------------------------------------------------
# This USED TO BE `cd "$(dirname "${BASH_SOURCE[0]}")"`, which is right when the
# script is executed directly (how --dry-run is used) and WRONG under Slurm:
# MEASURED in job 12876, sbatch copies the batch script into the scheduler's own
# spool and runs the copy, so inside the container BASH_SOURCE[0] is under
# /tmp/slurm/ and not the bind-mounted repo.  The symptom was a child process
# dying on
#     can't open file '/tmp/slurm/scripts/make_smoke_data.py'
# after the job had already waited in the queue.  Note that MODEL_DIR survived
# that run only because it is an absolute default rather than derived from here.
#
# Resolution order, and why:
#   1. ${WORKSPACE} -- the only channel that reaches the container.  MEASURED:
#      environment variables arrive intact (SMOKE=1 did), script arguments do
#      NOT (script args: []) and every SLURM_* variable is unset, so neither
#      argv nor SLURM_SUBMIT_DIR can be used to carry this.
#   2. the script's own directory, but ONLY if it actually looks like the repo.
#      This is what keeps `./finetune_chunk_slurm.sh --dry-run` working from a
#      checkout on a laptop, where /workspace does not exist.
#   3. /workspace, the #CONTAINER bind mount above (confirmed working: a probe
#      job listed the repo contents there).
# Hardcoding 3. unconditionally would break 2.; trusting 2. unconditionally is
# the bug being fixed.
WORKSPACE_FALLBACK="/workspace"

# Files that must exist in a real checkout.  scripts/make_smoke_data.py is the
# very file whose absence produced the failure above, so probing for it means
# the preflight fails on exactly the condition that used to reach a child
# process; model.py is a second, independent marker so that a directory holding
# a stray copy of the scripts/ tree cannot masquerade as the repo root.
WORKSPACE_MARKERS="scripts/make_smoke_data.py model.py"

workspace_missing_markers() {
    local root="$1" marker out=""
    # shellcheck disable=SC2086  # deliberate word splitting: the markers list is
    # a space-separated literal defined right above and contains no globs.
    for marker in ${WORKSPACE_MARKERS}; do
        [ -e "${root}/${marker}" ] || out="${out}${out:+ }${marker}"
    done
    printf '%s' "${out}"
}

# Absolute, symlink-resolved form of a directory; fails if it is not a directory.
abs_dir() { ( cd "$1" >/dev/null 2>&1 && pwd ); }

workspace_script_dir="$(abs_dir "$(dirname "${BASH_SOURCE[0]}")" || true)"

if [ -n "${WORKSPACE:-}" ]; then
    workspace="$(abs_dir "${WORKSPACE}" || printf '%s' "${WORKSPACE}")"
    workspace_source="env (WORKSPACE=${WORKSPACE})"
elif [ -n "${workspace_script_dir}" ] \
    && [ -z "$(workspace_missing_markers "${workspace_script_dir}")" ]; then
    workspace="${workspace_script_dir}"
    workspace_source="detected (directory of this script)"
else
    workspace="${WORKSPACE_FALLBACK}"
    workspace_source="fallback (script directory does not look like the repo)"
fi

# Validate here rather than letting a child process report a bare "No such file
# or directory" from inside a job that has already queued.  One line of
# diagnosis now is worth an hour of turnaround later.
workspace_missing="$(workspace_missing_markers "${workspace}")"
if [ -n "${workspace_missing}" ]; then
    {
        printf 'Error: the resolved workspace is not a SenseVoice checkout.\n'
        printf '  resolved:  %s\n' "${workspace}"
        printf '  resolved by: %s\n' "${workspace_source}"
        printf '  probed for: %s\n' "${WORKSPACE_MARKERS}"
        printf '  missing:    %s\n' "${workspace_missing}"
        printf '  looked at:\n'
        printf '    WORKSPACE env var:    %s\n' "${WORKSPACE:-<unset>}"
        printf '    script directory:     %s\n' "${workspace_script_dir:-<unresolvable>}"
        printf '    fallback:             %s\n' "${WORKSPACE_FALLBACK}"
        printf '\n'
        printf 'Under sbatch the batch script is executed from a copy in the scheduler spool\n'
        printf '(e.g. /tmp/slurm/...), so the script directory is NOT the repo and the repo can\n'
        printf 'only be found through the bind mount.  Fix: export WORKSPACE=/workspace (or\n'
        printf 'wherever "#CONTAINER -v ...:/workspace" puts the repo) in the job environment --\n'
        printf 'environment variables do reach the container; script arguments and SLURM_* do\n'
        printf 'not.  Running the script directly from a checkout needs no WORKSPACE at all.\n'
    } >&2
    exit 1
fi
info "workspace: ${workspace} [${workspace_source}]"

# ---------------------------------------------------------------------------
# Slurm environment (logging only -- see the header: all of it is expected to be
# UNSET inside the container, so every read is guarded and nothing below may
# depend on any of it)
# ---------------------------------------------------------------------------
info "slurm env (expected empty inside the container):"
info "  SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>} SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
info "  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>} SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-<unset>}"
info "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  host=$(hostname)"

# A stable id for log/hydra directory names.  Uses the job id when one happens to
# be present (e.g. when run outside the container) and a timestamp+pid otherwise.
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

# ---------------------------------------------------------------------------
# GPU count
# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES is the only trustworthy source here (measured: it is set,
# SLURM_* is not).  nvidia-smi is the fallback for a bare-metal/interactive run
# where it is unset; 1 is the last resort so the script degrades to single-GPU
# rather than dying.
derive_gpu_count() {
    local cvd="${CUDA_VISIBLE_DEVICES:-}" n=""

    if [ -n "${cvd}" ]; then
        n="$(printf '%s' "${cvd}" | awk -F, '{ c = 0; for (i = 1; i <= NF; i++) if (length($i)) c++; print c }')"
        if [ -n "${n}" ] && [ "${n}" -gt 0 ] 2>/dev/null; then
            printf '%s' "${n}"
            return 0
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
        if [ -n "${n}" ] && [ "${n}" -gt 0 ] 2>/dev/null; then
            printf '%s' "${n}"
            return 0
        fi
    fi

    printf '1'
}

# ---------------------------------------------------------------------------
# CONFIGURATION -- every value below is overridable from the environment
# ---------------------------------------------------------------------------

# --- paths (container-internal; see the #CONTAINER bind mounts above) ---
MODEL_DIR="${MODEL_DIR:-/workspace/models/SenseVoiceSmall}"
TRAIN_JSONL="${TRAIN_JSONL:-/corpus/vn/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/corpus/vn/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/chunk_cluster}"

# --- corpus-size expectation (opt-in; unset = the behaviour this script has
#     always had) ---
# WHY THIS GUARD EXISTS
# A run trained on the WRONG CORPUS SUCCEEDS, and nothing else here can see it.
# MEASURED: a job whose training manifest had been silently rebuilt over a
# 4-archive SUBSET trained on 16.5 audio-h where 813.8 were intended, finished in
# 4 minutes instead of 90, and reported
#     SENSEVOICE_JOB_OK rc=0 preflight=passed
# Every other check passed HONESTLY: the manifest parsed, every sampled clip
# existed and was readable, source_len was in the right units, the projected
# schedule was self-consistent with the corpus it was handed, the disk budget
# held, and the job-status sentinel was telling the truth -- the run genuinely
# succeeded, just at the wrong thing.  The sentinel cannot help with this class of
# failure by construction, and neither can sacct, the loss curve or the log.
#
# The one fact nobody had written down is how big the corpus was SUPPOSED to be:
# that is knowledge the caller has and the manifest does not, so there is nowhere
# to derive it from.  It is declared here instead, and checked in the preflight
# against the audio-hours that preflight already projects.
#
# Both expectations are OPT-IN and empty by default -- unset, the preflight is
# byte-for-byte the preflight it was, so existing invocations are unaffected.
#   EXPECT_TRAIN_HOURS            audio-hours the train manifest is believed to
#                                 hold (per epoch, i.e. the corpus itself)
#   EXPECT_TRAIN_HOURS_TOLERANCE  accepted fractional drift, 0.05 = +/-5%, so a
#                                 manifest that legitimately moves a little does
#                                 not block a run
#   EXPECT_VAL_CLIPS              exact clip count of the val manifest.  Exact
#                                 rather than toleranced on purpose: val is pinned
#                                 to a fixed set precisely so round-to-round
#                                 numbers are comparable, and a silently replaced
#                                 val invalidates every comparison while looking
#                                 perfectly healthy.
EXPECT_TRAIN_HOURS="${EXPECT_TRAIN_HOURS:-}"
EXPECT_TRAIN_HOURS_TOLERANCE="${EXPECT_TRAIN_HOURS_TOLERANCE:-0.05}"
EXPECT_VAL_CLIPS="${EXPECT_VAL_CLIPS:-}"

# --- device ---
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

# --- schedule ---
# 4 epochs, matching the MPS baseline so the two runs are comparable.
MAX_EPOCH="${MAX_EPOCH:-4}"

# --- learning rate ---
# 2e-4 is the DEFAULT CEILING, inherited from finetune_chunk.sh and
# finetune_chunk_mps.sh.  The check below refuses to start if LR exceeds it, and
# that refusal is the behaviour whenever LR_CEILING is left unset.
#
# The ceiling is overridable, but raising it must be a deliberate, explicit act:
# this is an adaptation of an already-trained checkpoint, and the failure mode to
# fear is catastrophic forgetting of full-attention behaviour, not underfitting.
# A raised ceiling is announced loudly on stderr and recorded in
# chunk_geometry.json, so which ceiling was in force is part of the run's
# identity when two runs are compared later.  If the per-epoch evaluation shows
# forgetting, LOWER LR (5e-5 or 1e-5 are the natural next steps) and re-run.
LR_CEILING_DEFAULT="0.0002"
LR_CEILING="${LR_CEILING:-${LR_CEILING_DEFAULT}}"
LR="${LR:-0.0002}"

# Left empty on purpose: inherit scheduler_conf.warmup_steps from the base
# model's config.yaml (25000).  The preflight prints the projected peak LR so
# this is visible rather than surprising.  Set to override.
WARMUP_STEPS="${WARMUP_STEPS:-}"

# --- batching ---
# 94 GB of VRAM per H100 NVL permits FAR more than 6000 tokens per step -- this
# is not a memory-derived number and nothing here is close to the card's limit.
# 6000 is kept because the baseline MPS run used it and the round-1 results have
# to stay comparable: raising it changes the effective batch size, hence the
# gradient noise scale and the LR that would be appropriate, and would make the
# two runs measure different things.  Raise it only as a deliberate, recorded
# decision (and expect to revisit LR when you do).
#
# Units: dataset_conf.batch_size is compared against the sampler's
# ``max_len_in_batch * len(batch)``, built from the jsonl's ``source_len`` field,
# which this repo writes in 10 ms frames -- 100 units per audio-second, so 6000
# is ~60 audio-seconds per step PER RANK.  The preflight verifies that
# convention against real audio headers and refuses to start if the corpus uses
# another one.
SOURCE_LEN_UNITS_PER_SECOND=100
BATCH_TOKENS="${BATCH_TOKENS:-6000}"
# Hard ceiling on examples per batch (funasr default is 200).  Held at the MPS
# baseline's value for the same comparability reason as BATCH_TOKENS.
MAX_SAMPLES_PER_STEP="${MAX_SAMPLES_PER_STEP:-12}"
# Length-bucketing window.  Larger = tighter length grouping = less padding.
# Matches both sibling scripts.
SORT_SIZE="${SORT_SIZE:-1024}"
# CPU budget: 2 ranks x 5 workers + 2 main processes = 12 = --cpus-per-task.
# Raising this without raising --cpus-per-task oversubscribes the cgroup and
# makes the dataloader slower, not faster.
NUM_WORKERS="${NUM_WORKERS:-5}"
ACCUM_GRAD="${ACCUM_GRAD:-1}"

# --- checkpointing ---
# funasr's Trainer.save_checkpoint prunes ``saved_ckpts`` down to
# keep_nbest_models and os.remove()s the loser.  12 keeps the 4 per-epoch
# checkpoints plus a healthy margin of mid-epoch ones for the forgetting
# evaluation, while staying inside the disk budget the preflight checks.
KEEP_NBEST="${KEEP_NBEST:-12}"
# Final checkpoint averaging writes an extra file; 1 makes it a no-op.  The point
# of this run is to pick a single epoch by evaluation, not to average.
AVG_NBEST="${AVG_NBEST:-1}"
# Mid-epoch safety checkpoints.  With a 24 h wall clock and a chained follow-up
# job, this is what bounds how much work a timeout throws away.  funasr asserts
# save_checkpoint_interval == validate_interval, so this single knob drives both.
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# Resume from OUTPUT_DIR/model.pt if it exists.  This is what makes the chained
# submission in scripts/submit_chunk_chain.sh work: a link that finds a finished
# run exits almost immediately, so surplus links are harmless.
RESUME="${RESUME:-true}"
# Approximate on-disk size of one SenseVoiceSmall checkpoint, used by the disk
# preflight.  Measured on the MPS run; override if the model changes.
CKPT_SIZE_GIB="${CKPT_SIZE_GIB:-2.7}"

# --- wall-clock ceiling ---
# The wall-clock limit this script BELIEVES the scheduler will enforce, in hours.
# See the PROVENANCE block at the top of this file: sacct records jobs 12899 and
# 12904 with Timelimit=08:00:00 even though this header requests --time=24:00:00,
# so the request is being clamped.  8 is that observation, not a value read from
# the QOS -- and it CANNOT be read at runtime, because SLURM_* is unset inside the
# container (see WHY THIS SCRIPT NEVER READS SLURM_JOB_ID).  The preflight
# compares the requested --time against this and warns on a mismatch; set it to
# whatever you have confirmed for your account, and the warning goes quiet.
TIME_CEILING_HOURS="${TIME_CEILING_HOURS:-8}"

# --- seed checkpoint and emotion-head knobs (opt-in; unset = the behaviour this
#     script has always had) ---
# All three are EMPTY by default and append nothing to the command when empty, so
# an invocation that sets none of them emits byte-for-byte the command it emitted
# before they existed.  Round-2 reproducibility depends on that.
#
# INIT_PARAM -- path to a checkpoint to initialise the weights from, i.e. start
# round N from a chosen epoch of round N-1 rather than from the stock model.
# VERIFIED against the installed funasr 1.4.1: bin/train_ds.py:105 builds the
# model through ``AutoModel(**kwargs)``, whose build_model reads this key at
# auto/auto_model.py:620-634 and hands it to train_utils/load_pretrained_model.py.
# That function documents the extended ``<file_path>:<src_key>:<dst_key>:<exclude_keys>``
# form, so a scoped load is available if a later round needs one.
#
# WHY THE PREFLIGHT CHECKS THE PATH
# auto_model.py:633-634 does NOT raise when the file is missing -- it print()s
# "error, init_param does not exist!" and CARRIES ON training from randomly
# initialised weights.  That line is one of thousands in a torchrun log, so a
# typo'd path costs a full GPU allocation and produces a checkpoint that looks
# ordinary and is worthless.  The preflight refuses the run instead.
#
# INIT_PARAM AND RESUME=true: RESUME WINS, AND THAT IS NOT A CONFLICT
# RESUME defaults to true, so INIT_PARAM is normally set alongside it and the
# combination is the NORMAL case, not a mistake -- failing on it would make
# INIT_PARAM unusable without also passing RESUME=false.  It is also already
# unambiguous in funasr: init_param is applied while the model is being built
# (train_ds.py:105), and Trainer.resume_checkpoint runs afterwards at
# train_ds.py:170, overwriting those weights -- but only ``if self.resume`` AND
# ${OUTPUT_DIR}/model.pt actually exists.  So:
#   * fresh OUTPUT_DIR (no model.pt) -> INIT_PARAM applies.  This is the intended
#     round-3 flow: seed from round-2 epoch 3, resume left at its default.
#   * OUTPUT_DIR already holds model.pt -> the resumed checkpoint wins and
#     INIT_PARAM is silently ignored, which is also correct: a job restarted by
#     the chain must not throw away its progress and re-seed.
# The preflight WARNS and names which of the two will supply the weights, because
# the difference is invisible in the log otherwise.  It never fails on it.
INIT_PARAM="${INIT_PARAM:-}"

# RICH_WEIGHT -- weight of the "rich" (emotion/event) loss term, appended as
# ++model_conf.rich_loss_weight.  Empty leaves the model's own default alone.
RICH_WEIGHT="${RICH_WEIGHT:-}"

# EMO_MASK_TOKEN_ID -- token id whose emotion slot is mapped to ignore_id, so the
# emotion position drops out of the loss instead of training against a label that
# is not really there.  The intended value is 24991, the id of the single token
# <|SER|> in ${MODEL_DIR}/chn_jpn_yue_eng_ko_spectok.bpe.model.
#
# ++model_conf.<key> is the right override path rather than a bare ++<key>:
# auto_model.py:615-618 merges kwargs["model_conf"] into the dict it then splats
# into the model constructor (``model_class(**model_conf)``), and
# SenseVoiceSmall.__init__ takes **kwargs, so both keys arrive as constructor
# kwargs.  The model's own config.yaml already carries a model_conf: block, which
# is what these overrides merge into.
EMO_MASK_TOKEN_ID="${EMO_MASK_TOKEN_ID:-}"

# --- precision ---
# See the bf16 resolution block further down: this is a REQUEST, not the final
# answer.  The script probes the installed funasr for a real bf16 option and
# falls back to fp32 with a loud warning rather than inventing a flag.
USE_BF16="${USE_BF16:-true}"

# --- logging ---
LOG_INTERVAL="${LOG_INTERVAL:-20}"

# --- smoke ---
SMOKE="${SMOKE:-0}"

# --- chunk geometry (must match finetune_chunk.sh and finetune_chunk_mps.sh) ---
# See docs/chunk_training.md: chunk_size is the TOTAL window width
# (pad_left + stride + pad_right), which is NOT the inference-side convention.
# Each entry is one dynamic-chunk operating point; one is drawn per step.
# An entry <= 0 is a full-attention sentinel and skips the geometry checks.
#
# This geometry is now QUADRUPLICATED across the repo -- here, in
# finetune_chunk.sh, in finetune_chunk_mps.sh, and in scripts/eval_chunk_gap.py's
# TRAINING_CHUNK_CONFIG -- and nothing links them.  The DEFAULT_* constants exist
# so that at least this file cannot drift from itself: they are both the defaults
# applied below and the reference the drift warning compares against.
DEFAULT_CHUNK_SIZES="8,12,16"
DEFAULT_STRIDES="6,10,14"
DEFAULT_PAD_LEFTS="0,0,0"
DEFAULT_LOOK_BACKS="1,1,1"

CHUNK_SIZES="${CHUNK_SIZES:-${DEFAULT_CHUNK_SIZES}}"
STRIDES="${STRIDES:-${DEFAULT_STRIDES}}"
PAD_LEFTS="${PAD_LEFTS:-${DEFAULT_PAD_LEFTS}}"
LOOK_BACKS="${LOOK_BACKS:-${DEFAULT_LOOK_BACKS}}"

# ---------------------------------------------------------------------------
# SMOKE mode
# ---------------------------------------------------------------------------
# Purpose: prove the CONTAINER works end to end -- image pulled, bind mounts
# present, CUDA visible to torch, funasr importable, chunk config accepted, a
# handful of optimizer steps actually run and a checkpoint lands on /outputs --
# in minutes rather than hours.  It says nothing about model quality.
#
# Unlike finetune_chunk.sh's smoke path this stays on train_ds.py and on the GPU:
# that script drops to bin/train.py only because train_ds.py cannot run on CPU,
# and running the real trainer is the entire point of a container check.  What is
# borrowed from it is the rest of the recipe -- locally generated data via
# scripts/make_smoke_data.py, batch_type=token with a small sort window, and
# intervals small enough that validation and checkpointing both fire.
#
# batch_type=example is NOT usable: SenseVoiceCTCDataset.__getitem__ reuses
# ``batch_size`` as a per-clip frame-length cap (``if speech_lengths >
# self.batch_size: continue``), so every short smoke clip would be silently
# dropped, the collator would emit its "data is empty!" 1x1 dummy batch, and
# model.py would die on text[:, 3] with an IndexError.  800 is above the longest
# generated clip, so nothing is dropped.
SMOKE_DATA_DIR=""
if [ "${SMOKE}" = "1" ]; then
    info "mode: SMOKE (1 GPU, a few steps, container end-to-end check)"

    OUTPUT_DIR="${OUTPUT_DIR_SMOKE:-${OUTPUT_DIR}_smoke}"
    # Generated under OUTPUT_DIR, not under the repo bind mount: /workspace may
    # be read-only or shared with other jobs, and /outputs is writable by
    # definition since the checkpoints go there.
    SMOKE_DATA_DIR="${OUTPUT_DIR}/smoke_data"
    TRAIN_JSONL="${SMOKE_DATA_DIR}/smoke_train.jsonl"
    VAL_JSONL="${SMOKE_DATA_DIR}/smoke_val.jsonl"
    # The smoke manifests are generated locally and have nothing to do with the
    # real corpus, so a corpus-size expectation inherited from the environment
    # would fail on data that is exactly right for this mode.  Dropped rather than
    # checked; the guard exists for TRAIN runs.
    if [ -n "${EXPECT_TRAIN_HOURS}" ] || [ -n "${EXPECT_VAL_CLIPS}" ]; then
        info "  ignoring EXPECT_TRAIN_HOURS/EXPECT_VAL_CLIPS: SMOKE uses generated data"
    fi
    EXPECT_TRAIN_HOURS=""
    EXPECT_VAL_CLIPS=""

    # RICH_WEIGHT and EMO_MASK_TOKEN_ID are deliberately KEPT in SMOKE mode.
    # Proving those two overrides is the entire reason to smoke-test this change:
    # the run must show acc_emo in the log and show that the ignore_id sentinel
    # does not blow up in the loss, and dropping them here would test the one
    # configuration nobody is about to run.
    #
    # INIT_PARAM is dropped.  Smoke runs against a handful of generated clips in a
    # throwaway OUTPUT_DIR to prove the container works; seeding it from a real
    # round's checkpoint tests nothing extra, and pulling a multi-GiB file over the
    # shared filesystem is the slowest thing in an otherwise minutes-long check.
    if [ -n "${INIT_PARAM}" ]; then
        info "  ignoring INIT_PARAM: SMOKE proves the container, not a training lineage"
    fi
    INIT_PARAM=""

    MAX_EPOCH=1
    LOG_INTERVAL=1
    BATCH_TOKENS=800
    MAX_SAMPLES_PER_STEP=4
    SORT_SIZE=4
    NUM_WORKERS=0
    SAVE_INTERVAL=2
    KEEP_NBEST=1
    AVG_NBEST=1
    RESUME=false
    # One GPU is enough to prove the container; two would only add a DDP
    # rendezvous to the list of things that can go wrong in a plumbing check.
    NPROC_PER_NODE=1
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
        export CUDA_VISIBLE_DEVICES
    fi
else
    info "mode: TRAIN (multi-GPU, torchrun + funasr bin/train_ds.py)"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(derive_gpu_count)}"
case "${NPROC_PER_NODE}" in
    ''|*[!0-9]*) die "NPROC_PER_NODE is not a positive integer: '${NPROC_PER_NODE}'" ;;
esac
[ "${NPROC_PER_NODE}" -ge 1 ] || die "NPROC_PER_NODE must be >= 1, got ${NPROC_PER_NODE}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    info "gpus:   ${NPROC_PER_NODE} (from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
else
    info "gpus:   ${NPROC_PER_NODE} (CUDA_VISIBLE_DEVICES unset; from nvidia-smi or the single-GPU fallback)"
fi

# ---------------------------------------------------------------------------
# Learning-rate ceiling
# ---------------------------------------------------------------------------
# With LR_CEILING unset the guard is exactly the fixed 2e-4 limit it has always
# been.  An overridden ceiling is recorded in chunk_geometry.json; a RAISED one
# is announced below as well, because it is the setting most likely to ruin a
# run without anything in the training log looking wrong.
LR_CEILING_OVERRIDDEN=0
if awk -v a="${LR_CEILING}" -v b="${LR_CEILING_DEFAULT}" 'BEGIN { exit !(a + 0 != b + 0) }'; then
    LR_CEILING_OVERRIDDEN=1
fi

if awk -v a="${LR}" -v b="${LR_CEILING}" 'BEGIN { exit !(a + 0 > b + 0) }'; then
    die "LR=${LR} exceeds the ceiling ${LR_CEILING}. Lowering it is fine; raising it is not."
fi

if awk -v a="${LR_CEILING}" -v b="${LR_CEILING_DEFAULT}" 'BEGIN { exit !(a + 0 > b + 0) }'; then
    cat >&2 <<EOF

***************************************************************************
*** WARNING: LEARNING-RATE CEILING RAISED ABOVE THE DEFAULT             ***
***************************************************************************
  ceiling in force: ${LR_CEILING}   (default: ${LR_CEILING_DEFAULT})
  this run trains at LR=${LR}

  The default ceiling exists because this is an adaptation of an
  already-trained checkpoint.  The failure mode to fear is CATASTROPHIC
  FORGETTING of full-attention behaviour, not underfitting -- a too-high LR
  does not diverge visibly, it quietly destroys what the base model already
  knew while the loss curve still looks plausible.

  If the per-epoch evaluation shows forgetting, LOWER LR (5e-5 or 1e-5 are the
  natural next steps) and re-run.  The ceiling in force is recorded in
  \${OUTPUT_DIR}/chunk_geometry.json so this run can be told apart from one
  trained under the default ceiling.
***************************************************************************

EOF
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

    local i c s p l pr v
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
# here does not merely go unnoticed by the evaluator: the evaluator will decode a
# geometry the model was never trained on and report the resulting CER as if it
# meant something.  Warn loudly rather than fail: a deliberate geometry sweep is
# a legitimate thing to want, it just cannot be evaluated without a matching edit
# on the other side.
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

  It also breaks comparability with the MPS baseline, which is the whole reason
  this run keeps BATCH_TOKENS=6000.

  Before evaluating, edit TRAINING_CHUNK_CONFIG in scripts/eval_chunk_gap.py to
  match the values above.  The resolved geometry is recorded in
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
# from whatever the hardcoded copies happen to say today.  Write it down next to
# the checkpoints instead.
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
    local ceiling_overridden="false"
    [ "${LR_CEILING_OVERRIDDEN}" = "1" ] && ceiling_overridden="true"

    # Written to a temp file and renamed so a reader racing the launch sees
    # either the previous run's record or this one's, never a half-written file.
    cat >"${tmp}" <<EOF
{
  "schema_version": 1,
  "written_by": "finetune_chunk_slurm.sh",
  "written_at": $(json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "host": $(json_string "$(hostname)"),
  "pid": $$,
  "run_id": $(json_string "${RUN_ID}"),
  "slurm_job_id": $(json_string "${SLURM_JOB_ID:-}"),
  "nproc_per_node": ${NPROC_PER_NODE},
  "output_dir": $(json_string "${OUTPUT_DIR}"),
  "model_dir": $(json_string "${MODEL_DIR}"),
  "device": $(json_string "${DEVICE}"),
  "bf16": ${BF16_EFFECTIVE},
  "batch_tokens": ${BATCH_TOKENS},
  "lr": $(awk -v x="${LR}" 'BEGIN { printf "%.10g", x + 0 }'),
  "lr_ceiling": $(awk -v x="${LR_CEILING}" 'BEGIN { printf "%.10g", x + 0 }'),
  "lr_ceiling_overridden": ${ceiling_overridden},
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
# PATH comes FIRST here, the opposite of finetune_chunk_mps.sh.  Inside the
# container the correct interpreter is the image's own python: NGC ships torch
# built against this image's exact CUDA/cuDNN/NCCL stack (see
# docker/Dockerfile.cluster).  The repo bind-mounted at /workspace may carry a
# .venv built on a developer's laptop, whose interpreter is a macOS binary and
# whose torch has no CUDA at all.  Picking that would either fail immediately or,
# worse, train on CPU at 1/50th speed for 24 h.  The venv is kept only as a
# fallback so --dry-run is usable outside the container.
python_has_torch() {
    [ -x "$1" ] && "$1" -c 'import torch' >/dev/null 2>&1
}

if [ -z "${PYTHON_BIN:-}" ]; then
    for _candidate in "$(command -v python3 2>/dev/null || true)" \
                      "$(command -v python 2>/dev/null || true)" \
                      "${workspace}/.venv/bin/python"; do
        if python_has_torch "${_candidate}"; then
            PYTHON_BIN="${_candidate}"
            break
        fi
    done
fi
if [ -z "${PYTHON_BIN:-}" ]; then
    # Nothing importable found; fall back to bare python3 so the preflight can
    # report *why* rather than the script dying with an unhelpful message.
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi
[ -n "${PYTHON_BIN}" ] || die "no python3 found on PATH and ${workspace}/.venv/bin/python does not exist"
[ -x "${PYTHON_BIN}" ] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
info "python: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# Reading a value back out of python
# ---------------------------------------------------------------------------
# NEVER capture a python subprocess's stdout wholesale.  MEASURED in the cluster
# image: importing funasr prints a banner to STDOUT at import time --
#
#     Notice: ffmpeg is not installed. torchaudio is used to load audio
#     If you want to use ffmpeg backend to load audio, please install it by:
#     ...
#
# from a module-level print() in funasr/utils/load_utils.py, guarded only by
# `if is_ffmpeg_installed()` -- which shells out to `ffmpeg -version`.  So the
# banner appears whenever ffmpeg is not resolvable on PATH, which is the case in
# this image; a developer whose laptop has ffmpeg installed sees nothing, which
# is why this stayed hidden until it ran on the cluster.  A plain
# "$(python -c '...print(x)')" then captures the banner *and* the value, and
# redirecting stderr cannot help because none of it is on stderr.
#
# Every value read back from python therefore goes through here: the payload tags
# its answer with a unique sentinel prefix, and only that line is extracted.  That
# stays correct whether the chatter arrives before the value, after it, or both --
# which `tail -1` would not, and a future funasr could print a banner after the
# value as easily as before it.
PY_VALUE_SENTINEL="__sv_py_value__"

# Runs the python payload supplied on stdin (which is handed the sentinel as
# argv[1]) and prints the first sentinel-tagged value it emitted, tag stripped.
# Fails if python failed or tagged nothing, so callers keep using `if ...; then`.
python_value() {
    local out
    # awk rather than `sed | head`: it consumes all of python's output instead of
    # exiting at the first match, so it cannot SIGPIPE the payload and trip
    # pipefail on what was actually a successful probe.
    out="$("${PYTHON_BIN}" - "${PY_VALUE_SENTINEL}" 2>/dev/null \
        | awk -v tag="${PY_VALUE_SENTINEL}" '
            index($0, tag) == 1 && !seen { seen = 1; print substr($0, length(tag) + 1) }
        ')" || return 1
    [ -n "${out}" ] || return 1
    printf '%s' "${out}"
}

# ---------------------------------------------------------------------------
# funasr trainer entry point
# ---------------------------------------------------------------------------
# bin/train_ds.py, resolved exactly the way finetune_chunk.sh's GPU path resolves
# it: the console-script directory first, the installed package second.
find_train_tool() {
    local funasr_bin funasr_pkg candidate
    if funasr_bin="$(command -v funasr 2>/dev/null)"; then
        candidate="$(dirname "${funasr_bin}")/train_ds.py"
        [ -f "${candidate}" ] && { printf '%s' "${candidate}"; return 0; }
    fi
    # The leading newline guarantees the sentinel starts a line even if the import
    # chatter ended without one.
    if funasr_pkg="$(python_value <<'FUNASR_PKG_PY'
import os
import sys

import funasr

sys.stdout.write("\n" + sys.argv[1] + os.path.dirname(funasr.__file__) + "\n")
FUNASR_PKG_PY
    )"; then
        candidate="${funasr_pkg}/bin/train_ds.py"
        [ -f "${candidate}" ] && { printf '%s' "${candidate}"; return 0; }
    fi
    return 1
}

if ! train_tool="$(find_train_tool)"; then
    check_fail "funasr's bin/train_ds.py not found. Is funasr installed in ${PYTHON_BIN}?"
    train_tool="<funasr>/bin/train_ds.py"
fi
info "trainer: ${train_tool}"

# ---------------------------------------------------------------------------
# bf16: verify the flag exists rather than inventing one
# ---------------------------------------------------------------------------
# VERIFIED against the funasr 1.4.1 actually installed for this project:
#   funasr/train_utils/trainer_ds.py  Trainer.__init__(..., use_bf16: bool = False)
#     -> self.dtype = torch.bfloat16
#     -> maybe_autocast(dtype=self.dtype) wraps the forward in
#        torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
#   funasr/bin/train_ds.py  Trainer(..., **kwargs.get("train_conf"))
#     -> so the hydra path is ++train_conf.use_bf16
# and, correctly, no GradScaler is built for bf16 (train_ds.py line ~167 gates it
# on trainer.use_fp16 alone) -- bf16 has fp32's exponent range and does not need
# loss scaling.
#
# The probe below re-checks that at run time instead of trusting this comment: if
# a future image ships a funasr whose Trainer has no use_bf16 parameter, hydra's
# ++ would happily create the key, Trainer would swallow it in **kwargs, and the
# run would silently train in fp32 while the log claimed bf16.  Fall back
# explicitly and say so.
funasr_bf16_support() {
    "${PYTHON_BIN}" - <<'BF16_PY'
import inspect
import sys

try:
    from funasr.train_utils.trainer_ds import Trainer
except Exception:
    sys.exit(2)

try:
    params = inspect.signature(Trainer.__init__).parameters
except (TypeError, ValueError):
    sys.exit(2)

sys.exit(0 if "use_bf16" in params else 1)
BF16_PY
}

BF16_EFFECTIVE="false"
case "${USE_BF16}" in
    true|True|1|yes)
        set +e
        funasr_bf16_support
        _bf16_rc=$?
        set -e
        case "${_bf16_rc}" in
            0)
                BF16_EFFECTIVE="true"
                info "precision: bf16 (++train_conf.use_bf16=true; verified present in the installed funasr Trainer)"
                ;;
            1)
                warn "bf16 was requested (USE_BF16=${USE_BF16}) but the installed funasr exposes no use_bf16 option on its Trainer. Falling back to fp32. Do NOT add a flag by hand: hydra's ++ would create a key nothing reads and the log would claim a precision the run is not using."
                info "precision: fp32 (bf16 requested but unavailable)"
                ;;
            *)
                warn "could not import funasr to verify bf16 support; assuming the funasr 1.4.1 behaviour (++train_conf.use_bf16 is real). The preflight below will fail on the same import if funasr is genuinely missing."
                BF16_EFFECTIVE="true"
                info "precision: bf16 (unverified)"
                ;;
        esac
        ;;
    *)
        info "precision: fp32 (USE_BF16=${USE_BF16})"
        ;;
esac

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
# A local directory is required rather than a hub id: funasr pip-installs the
# requirements.txt bundled inside a downloaded model directory, which can
# override this image's pinned numerical stack -- the exact swap
# docker/Dockerfile.cluster exists to prevent.  Compute nodes may also have no
# outbound network at all.
if [ ! -d "${MODEL_DIR}" ]; then
    check_fail "base model directory not found: ${MODEL_DIR} (see README 'Using a local model directory'; it must be staged on the shared filesystem before submitting)"
elif [ ! -f "${MODEL_DIR}/model.pt" ] || [ ! -f "${MODEL_DIR}/config.yaml" ]; then
    check_fail "base model directory ${MODEL_DIR} is missing model.pt and/or config.yaml"
fi
info "model: ${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Smoke data generation
# ---------------------------------------------------------------------------
if [ "${SMOKE}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    info "generating smoke data under ${SMOKE_DATA_DIR} ..."
    mkdir -p "${SMOKE_DATA_DIR}"
    "${PYTHON_BIN}" "${workspace}/scripts/make_smoke_data.py" \
        --wav-dir "${SMOKE_DATA_DIR}/wav" \
        --train-jsonl "${TRAIN_JSONL}" \
        --val-jsonl "${VAL_JSONL}"
fi

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# --dry-run deliberately does not generate the smoke corpus (it must not write
# anything), so under SMOKE=1 --dry-run these manifests are expected to be
# missing and the hint would otherwise send the reader after the wrong problem.
if [ "${SMOKE}" = "1" ]; then
    data_hint="generated by scripts/make_smoke_data.py at launch; --dry-run skips that step, so this is expected here"
else
    data_hint="stage the VN corpus on the shared filesystem bind-mounted at /corpus before submitting"
fi

for data_file in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
    if [ ! -f "${data_file}" ]; then
        check_fail "data manifest not found: ${data_file} (${data_hint})"
    elif [ ! -s "${data_file}" ]; then
        check_fail "data manifest is empty: ${data_file}"
    fi
done

# ---------------------------------------------------------------------------
# Preflight: GPUs, corpus sanity, projected schedule, disk budget
# ---------------------------------------------------------------------------
# Cluster turnaround is measured in hours, so everything that can be checked
# before the first optimizer step is checked here rather than discovered at
# hour 6.  nvidia-smi first (it answers "what hardware did the scheduler
# actually give me"), then one python subprocess for everything else.
info "preflight:"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia_smi_out="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null)" && [ -n "${nvidia_smi_out}" ]; then
        printf '%s\n' "${nvidia_smi_out}" | sed 's/^/  gpu: /'
    else
        check_fail "nvidia-smi is present but reported no GPUs; the container was launched without a GPU allocation"
    fi
else
    check_fail "nvidia-smi not found; this job is not seeing a GPU node (the container may have been launched without --gpus-per-task)"
fi

# The wall clock this job ASKED for, read back out of the #SBATCH header.
#
# This is the request, not the grant, and the two are known to differ here (see
# the PROVENANCE block at the top).  The grant is unreadable from inside: the site
# wrapper forwards no SLURM_* variables, so there is no job id to hand `scontrol`
# and no SLURM_TIMELIMIT to read.  The request is therefore the only end of the
# comparison this script can observe, which is exactly why the check below
# compares it against a RECORDED OBSERVATION rather than against the live limit.
#
# Reads ${BASH_SOURCE[0]} for the same reason site_header_check does: under sbatch
# that is the scheduler's spool copy, i.e. the exact text the wrapper parsed.
requested_walltime() {
    local src="${BASH_SOURCE[0]}" line=""
    [ -r "${src}" ] || return 0
    # `|| true` because grep exits 1 on no match, which is a state the caller
    # reports rather than an error that should trip set -e.
    line="$(grep -E '^#SBATCH --time=' "${src}" 2>/dev/null | head -n 1 || true)"
    printf '%s' "${line#\#SBATCH --time=}"
}

REQUESTED_WALLTIME="$(requested_walltime)"

run_preflight() {
    "${PYTHON_BIN}" - \
        "${DEVICE}" \
        "${TRAIN_JSONL}" \
        "${VAL_JSONL}" \
        "${MODEL_DIR}" \
        "${MAX_EPOCH}" \
        "${LR}" \
        "${WARMUP_STEPS}" \
        "${BATCH_TOKENS}" \
        "${MAX_SAMPLES_PER_STEP}" \
        "${SOURCE_LEN_UNITS_PER_SECOND}" \
        "${NPROC_PER_NODE}" \
        "${OUTPUT_DIR}" \
        "${KEEP_NBEST}" \
        "${CKPT_SIZE_GIB}" \
        "${BF16_EFFECTIVE}" \
        "${EXPECT_TRAIN_HOURS}" \
        "${EXPECT_TRAIN_HOURS_TOLERANCE}" \
        "${EXPECT_VAL_CLIPS}" \
        "${INIT_PARAM}" \
        "${RICH_WEIGHT}" \
        "${EMO_MASK_TOKEN_ID}" \
        "${RESUME}" \
        "${SMOKE}" \
        "${REQUESTED_WALLTIME}" \
        "${TIME_CEILING_HOURS}" \
        <<'PREFLIGHT_PY'
import json
import math
import os
import shutil
import sys

(
    device, train_jsonl, val_jsonl, model_dir, max_epoch, lr, warmup_arg,
    batch_tokens, samp_cap, units_per_sec, nproc, output_dir, keep_nbest,
    ckpt_gib, bf16, expect_train_hours, expect_train_tol, expect_val_clips,
    init_param, rich_weight, emo_mask_token_id, resume, smoke,
    requested_walltime, time_ceiling_hours,
) = sys.argv[1:26]

max_epoch = int(max_epoch)
lr = float(lr)
batch_tokens = int(batch_tokens)
samp_cap = int(samp_cap)
units_per_sec = float(units_per_sec)
nproc = int(nproc)
keep_nbest = int(keep_nbest)
ckpt_gib = float(ckpt_gib)
bf16 = bf16 == "true"

errors = []
notes = []

# The emotion-slot sentinel as it appears in a manifest's emo_target field.  Its
# token id is 24991, which is what EMO_MASK_TOKEN_ID carries; this is the string
# side of the same thing, and the manifest only ever holds the string.
EMO_SENTINEL = "<|SER|>"


def fail(msg):
    errors.append(msg)


# --- seed checkpoint and emotion-head knobs (opt-in) -----------------------
# All three are skipped entirely when empty, so an invocation that sets none of
# them gets byte-for-byte the preflight it got before they existed.
resume_on = resume == "true"
smoke_on = smoke == "1"

# Parsed here rather than beside its own check because the emo_target constancy
# check inside inspect() has to know whether emotion masking was requested.
emo_mask_id = None
if emo_mask_token_id.strip():
    try:
        emo_mask_id = int(emo_mask_token_id)
    except ValueError:
        fail(
            "EMO_MASK_TOKEN_ID must be a non-negative integer token id, got "
            f"{emo_mask_token_id!r}. The intended value is 24991, the id of the "
            "single token <|SER|> in the model's "
            "chn_jpn_yue_eng_ko_spectok.bpe.model."
        )
    else:
        if emo_mask_id < 0:
            fail(
                "EMO_MASK_TOKEN_ID must be a non-negative integer token id, got "
                f"{emo_mask_token_id!r}."
            )
            emo_mask_id = None
        else:
            print(f"  emotion masking: model_conf.emo_mask_token_id={emo_mask_id}")

if rich_weight.strip():
    rich_value = None
    try:
        rich_value = float(rich_weight)
    except ValueError:
        pass
    if rich_value is None or math.isnan(rich_value) or math.isinf(rich_value):
        fail(f"RICH_WEIGHT must be a non-negative number, got {rich_weight!r}")
    elif rich_value < 0:
        fail(f"RICH_WEIGHT must be a non-negative number, got {rich_weight!r}")
    else:
        print(f"  rich loss: model_conf.rich_loss_weight={rich_value}")

# A missing init_param is NOT fatal inside funasr: auto_model.py:633-634 prints
# "error, init_param does not exist!" and trains on from randomly initialised
# weights.  One line in a torchrun log, a whole GPU allocation, and a checkpoint
# that looks ordinary and is worthless.  Caught here instead, before the queue.
if init_param.strip():
    if not os.path.isfile(init_param):
        fail(
            f"INIT_PARAM does not exist or is not a file: {init_param}. "
            "funasr does NOT fail on this -- it prints one line and trains from "
            "scratch instead -- so a typo here silently costs the whole run. "
            "The path must be container-internal (under the #CONTAINER bind "
            "mounts), not a login-node path."
        )
    elif not os.access(init_param, os.R_OK):
        fail(f"INIT_PARAM exists but is not readable: {init_param}")
    else:
        size_gib = os.path.getsize(init_param) / 1024**3
        print(f"  init_param: {init_param} ({size_gib:.2f} GiB)")
        # Not a failure: RESUME defaults to true, so this combination is the
        # normal way INIT_PARAM is used.  See the INIT_PARAM block in the
        # CONFIGURATION section -- init_param is applied when the model is built
        # and resume_checkpoint overwrites it afterwards, but only if a
        # checkpoint is actually there.  Say which one wins, because nothing in
        # the training log distinguishes the two.
        if resume_on:
            resume_ckpt = os.path.join(output_dir, "model.pt")
            if os.path.isfile(resume_ckpt):
                notes.append(
                    f"INIT_PARAM is set but RESUME=true and {resume_ckpt} already "
                    "exists, so the RESUMED checkpoint wins and INIT_PARAM will be "
                    "ignored (init_param is applied at model build, then "
                    "Trainer.resume_checkpoint overwrites it). This is correct for "
                    "a restarted/chained job. To seed from INIT_PARAM instead, "
                    "point OUTPUT_DIR at a fresh directory or pass RESUME=false."
                )
            else:
                notes.append(
                    f"INIT_PARAM is set with RESUME=true, but no {resume_ckpt} "
                    "exists yet, so the weights come from INIT_PARAM. A later "
                    "restart against this same OUTPUT_DIR will resume instead."
                )


# --- torch / CUDA ----------------------------------------------------------
try:
    import torch
except Exception as exc:  # pragma: no cover - environment problem
    print(f"  torch: NOT IMPORTABLE ({exc})")
    sys.exit(1)

print(f"  torch: {torch.__version__} (cuda build {torch.version.cuda})")
if device.startswith("cuda"):
    if not torch.cuda.is_available():
        # The single most expensive failure mode on this cluster: a CPU-only
        # wheel replaced the NGC build, the job runs, and 24 h later there is
        # nothing to show for it.
        fail(
            "torch.cuda.is_available() is False. Either the container got no GPU "
            "(check --gpus-per-task and CUDA_VISIBLE_DEVICES) or a CPU-only torch "
            "wheel replaced the NGC build (check PIP_CONSTRAINT in the image)."
        )
    else:
        count = torch.cuda.device_count()
        print(f"  cuda: {count} device(s) visible, {nproc} requested")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            print(
                f"  cuda:{i}: {props.name}, "
                f"{props.total_memory / 1024**2:.0f} MiB, sm_{props.major}{props.minor}"
            )
        if count < nproc:
            fail(
                f"{nproc} ranks requested but torch sees only {count} GPU(s); "
                "torchrun would start ranks with no device of their own"
            )
        elif count > nproc:
            notes.append(
                f"{count} GPUs visible but only {nproc} will be used; "
                "the rest are idle inside this allocation"
            )
        if bf16:
            supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            print(f"  bf16: torch.cuda.is_bf16_supported()={supported}")
            if not supported:
                fail(
                    "bf16 is enabled but this GPU/torch reports no bf16 support; "
                    "set USE_BF16=false"
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

    # --- constant emo_target -----------------------------------------------
    # THE ROUND-1/2 DEFECT, WHICH NOTHING IN THE PIPELINE CAUGHT.
    # Every clip was stamped <|NEUTRAL|>, so the emotion head trained against a
    # constant, acc_rich saturated at 1.0 by step 880, and the run looked like it
    # was learning beautifully.  A constant label is not visible to any other
    # check here: the manifest is well-formed, the audio is present, the hours are
    # right, the loss goes down.  It is only visible by looking at the label
    # distribution, which is what this does.
    #
    # emo_target is absent from older manifests; that is not the defect and says
    # nothing.  Only a field that is PRESENT and never varies is reported.
    emo_values = [r["emo_target"] for r in records if "emo_target" in r]
    if emo_values:
        # One pass, counted EXACTLY over every record -- not the 20-record sample
        # the audio and source_len checks below use.  The sentinel check depends
        # on that: "this manifest contains no <|SER|> anywhere" is not a claim a
        # sample can support, and it gates a fatal branch.
        emo_counts = {}
        for value in emo_values:
            emo_counts[value] = emo_counts.get(value, 0) + 1
        distinct = set(emo_counts)
        sentinel_n = emo_counts.get(EMO_SENTINEL, 0)
        top = max(emo_counts, key=emo_counts.get)
        print(
            f"  {label}: emo_target {len(distinct)} distinct value(s) over "
            f"{len(emo_values)}/{len(records)} records, most common {top!r} "
            f"{100.0 * emo_counts[top] / len(emo_values):.1f}%; {EMO_SENTINEL} "
            f"sentinel {sentinel_n} ({100.0 * sentinel_n / len(emo_values):.1f}%)"
        )

        if len(distinct) == 1:
            only = next(iter(distinct))
            coverage = (
                "100% of"
                if len(emo_values) == len(records)
                else f"{len(emo_values)}/{len(records)} of"
            )
            msg = (
                f"{label}: emo_target is the single value {only!r} across "
                f"{coverage} {len(records)} records -- the emotion label carries no "
                "information and an emotion head trained on it will report a "
                "near-perfect acc_rich while having learned nothing. This is "
                "exactly the round-1/2 defect. Re-label the manifest (see "
                "scripts/label_emotions_*.py) or mask the emotion slot out of the "
                "loss with EMO_MASK_TOKEN_ID."
            )
            # Fatal only when emotion masking was actually requested, and only
            # outside SMOKE.  Asking to mask the emotion slot against a manifest
            # whose emotion labels are constant is a contradiction: the run is
            # either pointed at the wrong manifest or the labelling step never
            # ran, and either way it is not the experiment that was intended.
            #
            # SMOKE is exempt on purpose.  A DEFAULT smoke corpus is constant by
            # construction -- scripts/make_smoke_data.py gives every generated clip
            # the same emo_target unless it is asked for a mixture -- so failing
            # here would block the one run that exists to prove these overrides
            # load at all.  On throwaway plumbing data a degenerate label
            # distribution is expected and harmless.
            #
            # This is an exemption for the DEFAULT smoke corpus specifically, not a
            # claim that smoke manifests can never vary: make_smoke_data.py can be
            # asked for a realistic mixture, and such a run simply does not trip
            # this branch.  Do not widen the exemption to the sentinel check below
            # on the strength of this one -- see the reasoning there.
            if emo_mask_id is not None and not smoke_on:
                fail(msg)
            else:
                notes.append(msg)

        # --- sentinel / EMO_MASK_TOKEN_ID agreement -------------------------
        # These two settings are chosen in two different places -- the manifest
        # by scripts/prepare_vn_data.py --emo-labels, the mask by the sbatch
        # environment -- and nothing else connects them.
        #
        # model.py raises RuntimeError from forward() when it meets the sentinel
        # with no emo_mask_token_id configured, which is the right guard but fires
        # at TRAINING STEP 1: after the queue wait, the GPU allocation, the
        # dataloader build and the checkpoint load.  Everything needed to catch it
        # is already in this manifest, so catch it here for free instead.
        #
        # NOT exempt under SMOKE, unlike the constant-label check above, and the
        # difference is deliberate: a constant label is a DATA-QUALITY judgement
        # that is meaningless on generated throwaway clips, whereas this is a hard
        # runtime fact -- the forward pass raises just as reliably on smoke data as
        # on the real corpus, so exempting it would only move the same crash a few
        # minutes later.  scripts/make_smoke_data.py can emit a realistic sentinel
        # mixture precisely so the smoke run exercises this path; a smoke run that
        # asks for that mixture and then forgets EMO_MASK_TOKEN_ID has proved
        # nothing, and must not be allowed to look like it did.
        #
        # Applied to the val manifest as well as train: Trainer.validate_epoch
        # calls the same forward_step (trainer_ds.py:849), so a sentinel in val
        # raises identically, just at the first validation instead of step 1.
        if sentinel_n and emo_mask_id is None:
            fail(
                f"{label}: {sentinel_n} record(s) "
                f"({100.0 * sentinel_n / len(emo_values):.1f}%) carry the "
                f"{EMO_SENTINEL} emotion sentinel but EMO_MASK_TOKEN_ID is not "
                "set, so the emotion slot would be trained AGAINST the sentinel "
                "instead of masked out of the loss. model.py raises RuntimeError "
                "on this at training step 1, after the allocation is already "
                "spent. Fix whichever matches what you meant to run: set "
                "EMO_MASK_TOKEN_ID=24991 (the id of "
                f"{EMO_SENTINEL}) to drop the emotion slot from the loss, or "
                "rebuild the manifest WITHOUT scripts/prepare_vn_data.py "
                "--emo-labels so it carries real emotion labels."
            )
        elif emo_mask_id is not None and sentinel_n == 0:
            notes.append(
                f"{label}: EMO_MASK_TOKEN_ID={emo_mask_id} is set but not one of "
                f"{len(emo_values)} records carries the {EMO_SENTINEL} sentinel, "
                "so the mask will never fire and the emotion head trains on the "
                "labels as they are. MOST LIKELY CAUSE: this manifest was built "
                "without scripts/prepare_vn_data.py --emo-labels, i.e. it is the "
                "round-2 constant-<|NEUTRAL|> corpus, which would silently "
                "reproduce the exact defect round 3 exists to repair. Legitimate "
                "only if this is a deliberate ablation."
            )

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

    # The corpus must actually be on the shared filesystem the container sees.
    # Sample rather than stat tens of thousands of files over NFS.
    step = max(1, len(records) // 20)
    sample = records[::step][:20]
    absent = [r["source"] for r in sample if not os.path.isfile(r["source"])]
    if absent:
        fail(
            f"{label}: {len(absent)}/{len(sample)} sampled audio files do not exist "
            f"(e.g. {absent[0]}); check the #CONTAINER bind mounts and that the "
            "manifest paths are container-internal, not login-node paths"
        )
    else:
        empty = [r["source"] for r in sample if os.path.getsize(r["source"]) == 0]
        if empty:
            fail(f"{label}: sampled audio file is zero bytes: {empty[0]}")
        else:
            print(f"  {label}: {len(sample)}/{len(sample)} sampled audio files present")

    # Verify the source_len convention against real audio headers.  The batch
    # budget is expressed in these units, so getting it wrong moves the step size
    # by 6x and silently destroys comparability with the MPS baseline.
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
                    "Fix the manifest or set BATCH_TOKENS explicitly."
                )
    except ImportError:
        notes.append("soundfile unavailable: source_len convention not verified")
    except Exception as exc:
        notes.append(f"{label}: could not verify source_len convention: {exc}")

    return len(records), sum(seconds), sum(lens)


train = inspect(train_jsonl, "train")
val = inspect(val_jsonl, "val")

# --- projected schedule ----------------------------------------------------
epoch_hours = None
if train:
    n_clips, total_seconds, total_units = train
    # DDP splits the corpus across ranks, so each rank sees 1/nproc of it and the
    # optimizer-step count drops accordingly.  The *effective* batch is
    # nproc x batch_tokens, which is the number that matters when comparing this
    # run against the single-device MPS baseline.
    per_rank_units = total_units / nproc
    per_rank_clips = n_clips / nproc
    steps_per_epoch = max(
        math.ceil(per_rank_units / batch_tokens),
        math.ceil(per_rank_clips / samp_cap),
    )
    total_steps = steps_per_epoch * max_epoch
    print(
        f"  projected: ~{steps_per_epoch} steps/epoch, ~{total_steps} steps total "
        f"({nproc} rank(s) x batch_size={batch_tokens} units "
        f"= ~{nproc * batch_tokens / units_per_sec:.0f} effective audio-s/step, "
        f"cap {samp_cap} clips/rank)"
    )
    # Bound once and reused by the corpus-size expectation below, so the guard
    # can never disagree with the number printed here.
    epoch_hours = total_seconds / 3600.0
    print(
        f"  projected: {epoch_hours:.2f} audio-h per epoch, "
        f"{epoch_hours * max_epoch:.2f} audio-h total"
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

# --- corpus-size expectation (opt-in) --------------------------------------
# The only check here that compares the corpus against something OUTSIDE it.
# Everything above asks "is this manifest internally consistent and readable",
# which a manifest rebuilt over a subset of the archives passes perfectly -- it is
# a valid corpus, just not the intended one, and the resulting run succeeds.  See
# the WHY block next to EXPECT_TRAIN_HOURS in this file's CONFIGURATION section.
# Unset expectations are skipped entirely, so the default behaviour is unchanged.
train_clips = train[0] if train else None
val_clips = val[0] if val else None

if expect_train_hours.strip():
    expected_h = None
    try:
        expected_h = float(expect_train_hours)
    except ValueError:
        pass
    if expected_h is None or not (expected_h > 0) or math.isinf(expected_h):
        fail(f"EXPECT_TRAIN_HOURS must be a positive number of hours, got {expect_train_hours!r}")
        expected_h = None

    tol = None
    try:
        tol = float(expect_train_tol)
    except ValueError:
        pass
    if tol is None or not 0.0 <= tol < 1.0:
        fail(
            "EXPECT_TRAIN_HOURS_TOLERANCE must be a fraction in [0, 1) "
            f"(0.05 = +/-5%), got {expect_train_tol!r}"
        )
        tol = None

    # epoch_hours is None only when the manifest itself already failed above; that
    # failure is the one to report, not a comparison against nothing.
    if expected_h is not None and tol is not None and epoch_hours is not None:
        low = expected_h * (1.0 - tol)
        high = expected_h * (1.0 + tol)
        if low <= epoch_hours <= high:
            print(
                f"  corpus size: train {epoch_hours:.2f} audio-h matches "
                f"EXPECT_TRAIN_HOURS={expected_h:.2f} "
                f"(+/-{tol * 100:.1f}%, {train_clips} clips)"
            )
        else:
            fail(
                f"train manifest holds {epoch_hours:.2f} audio-h but "
                f"EXPECT_TRAIN_HOURS={expected_h:.2f} "
                f"(tolerance +/-{tol * 100:.1f}%, accepted {low:.2f}-{high:.2f} h); "
                f"manifests as found: train {train_clips} clips in {train_jsonl}, "
                f"val {val_clips} clips in {val_jsonl}. "
                "MOST LIKELY CAUSE: a data-prep run rebuilt this manifest over a "
                "SUBSET of the archives and replaced the full one. "
                "Re-run data prep over the full archive set, or pass the corpus "
                "size you actually intend as EXPECT_TRAIN_HOURS. This is the only "
                "check that can see it: a run on the wrong corpus trains, "
                "converges and exits 0."
            )

if expect_val_clips.strip():
    expected_val = None
    try:
        expected_val = int(expect_val_clips)
    except ValueError:
        pass
    if expected_val is None or expected_val <= 0:
        fail(f"EXPECT_VAL_CLIPS must be a positive integer clip count, got {expect_val_clips!r}")
    elif val_clips is not None:
        if val_clips == expected_val:
            print(f"  corpus size: val {val_clips} clips matches EXPECT_VAL_CLIPS exactly")
        else:
            fail(
                f"val manifest holds {val_clips} clips but EXPECT_VAL_CLIPS="
                f"{expected_val} (exact match required, no tolerance: val is "
                "pinned to a fixed set so round-to-round numbers stay comparable, "
                "and a replaced val invalidates every comparison while looking "
                f"perfectly healthy); manifests as found: train {train_clips} "
                f"clips in {train_jsonl}, val {val_clips} clips in {val_jsonl}. "
                "MOST LIKELY CAUSE: a data-prep run rebuilt this manifest over a "
                "SUBSET of the archives and replaced the pinned one."
            )

# --- disk budget -----------------------------------------------------------
# keep_nbest_models bounds how many checkpoints survive pruning, so the steady
# state is keep_nbest x one checkpoint, plus the averaged model and optimizer
# state that ride along.  Running /outputs out of space mid-run corrupts the
# checkpoint being written and takes the run with it, so this is a hard failure.
probe = os.path.abspath(output_dir)
while not os.path.isdir(probe):
    parent = os.path.dirname(probe)
    if parent == probe:
        break
    probe = parent
try:
    usage = shutil.disk_usage(probe)
    free_gib = usage.free / 1024**3
    required_gib = keep_nbest * ckpt_gib
    print(
        f"  disk: {free_gib:.1f} GiB free on {probe}, "
        f"~{required_gib:.1f} GiB projected "
        f"(KEEP_NBEST={keep_nbest} x {ckpt_gib} GiB)"
    )
    if free_gib < required_gib:
        fail(
            f"insufficient free space under {probe}: {free_gib:.1f} GiB free but "
            f"~{required_gib:.1f} GiB of checkpoints projected. Lower KEEP_NBEST, "
            "point OUTPUT_DIR at a larger filesystem, or clear old runs."
        )
    elif free_gib < required_gib * 1.5:
        notes.append(
            f"only {free_gib:.1f} GiB free for ~{required_gib:.1f} GiB of checkpoints; "
            "there is little headroom for logs or a concurrent job"
        )
except Exception as exc:
    notes.append(f"could not check free space on {probe}: {exc}")


# --- wall clock ------------------------------------------------------------
# A run cut off at the wall clock is the failure this script is least able to
# see. It leaves a COMPLETED job with a missing final epoch, and the site's
# scheduler discards the batch script's exit code, so sacct cannot tell it from a
# clean finish. The job-status sentinel cannot help either: the process is killed,
# so nothing gets the chance to write SENSEVOICE_JOB_FAILED.
#
# WHAT THIS CAN AND CANNOT CHECK
# It compares the wall clock this job REQUESTED against a RECORDED OBSERVATION of
# what the scheduler granted last time. It does NOT read the enforced limit: that
# is unreadable from inside the container (SLURM_* is unset, so there is no job id
# for scontrol), so this is a comparison against a number a human wrote down, and
# it is stale the moment the QOS changes. Re-measure with the commands in the
# PROVENANCE block at the top of this file if it disagrees with reality.
#
# WARNS RATHER THAN FAILS, deliberately. The ceiling is an observation of two
# jobs, not a read of enforced configuration; the truncation it predicts is fully
# recoverable through the resume chain; and the --time line cannot be templated
# from the environment, so a fatal check would block every run behind a manual
# edit of a site-specific line on evidence this file itself labels as observed
# rather than enforced.
def parse_slurm_duration(text):
    """Slurm accepts M, M:S, H:M:S, D-H, D-H:M and D-H:M:S. Returns hours."""
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        head, _, text = text.partition("-")
        try:
            days = int(head)
        except ValueError:
            return None
        if not text:
            return days * 24.0
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if days or len(parts) == 3:
        # D-H[:M[:S]] and H:M:S both count from the left.
        nums += [0] * (3 - len(nums))
        h, m, s = nums[:3]
    elif len(parts) == 2:
        h, m, s = 0, nums[0], nums[1]  # M:S
    else:
        h, m, s = 0, nums[0], 0  # bare minutes
    return days * 24.0 + h + m / 60.0 + s / 3600.0

ceiling_h = None
try:
    ceiling_h = float(time_ceiling_hours)
except ValueError:
    pass
if ceiling_h is None or not ceiling_h > 0:
    fail(f"TIME_CEILING_HOURS must be a positive number of hours, got {time_ceiling_hours!r}")
elif not requested_walltime.strip():
    notes.append(
        "could not read '#SBATCH --time=' from this script, so the requested wall "
        f"clock could not be checked against the {ceiling_h:g} h ceiling. If this "
        "run is long, submit it through scripts/submit_chunk_chain.sh so it "
        "survives being cut off."
    )
else:
    requested_h = parse_slurm_duration(requested_walltime)
    if requested_h is None:
        notes.append(
            f"could not parse '#SBATCH --time={requested_walltime}', so the "
            f"requested wall clock could not be checked against the "
            f"{ceiling_h:g} h ceiling"
        )
    elif requested_h > ceiling_h:
        notes.append(
            f"WALL CLOCK: this job requests --time={requested_walltime} "
            f"({requested_h:g} h) but the scheduler is believed to grant only "
            f"{ceiling_h:g} h (TIME_CEILING_HOURS), so the request is expected to "
            f"be CLAMPED and the job killed at ~{ceiling_h:g} h. That leaves a "
            "COMPLETED job with a missing final epoch: the exit code is discarded "
            "by this site's scheduler, and a killed process never gets to write "
            "the job-status sentinel, so NOTHING downstream can tell it from a "
            "clean finish. MITIGATION: submit through "
            "scripts/submit_chunk_chain.sh, which chains links with "
            "--dependency=afterany so each one resumes from "
            f"{os.path.join(output_dir, 'model.pt')}; a single job is only safe if "
            f"it finishes inside {ceiling_h:g} h. This compares the REQUEST against "
            "a recorded observation -- the enforced limit cannot be read from "
            "inside the container (SLURM_* is unset). If it is wrong, re-measure "
            "and set TIME_CEILING_HOURS."
        )
    else:
        print(
            f"  wall clock: requested --time={requested_walltime} "
            f"({requested_h:g} h), within the believed {ceiling_h:g} h ceiling"
        )

for note in notes:
    print(f"  note: {note}")
for err in errors:
    print(f"  FAIL: {err}")

sys.exit(1 if errors else 0)
PREFLIGHT_PY
}

if run_preflight; then
    PREFLIGHT_STATUS="passed"
else
    # Recorded before check_fail, which exits on a real run: the status sentinel
    # must be able to say the checks are what stopped the job.
    PREFLIGHT_STATUS="failed"
    check_fail "preflight checks failed (see FAIL lines above)"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_ASSIGNMENTS=()

# Without this the log stalls in the pipe to tee and `tail -f` on the Slurm
# output file shows nothing for minutes at a time -- which, on a machine you can
# only reach through squeue, is indistinguishable from a hung job.
export PYTHONUNBUFFERED=1
ENV_ASSIGNMENTS+=("PYTHONUNBUFFERED=1")

# torchrun sets OMP_NUM_THREADS=1 itself (with a warning) when it is unset, but
# doing it here makes the CPU budget explicit: 12 CPUs are already fully
# allocated to 2 ranks x 5 dataloader workers + 2 main processes, so letting
# OpenMP spawn a thread per core inside each of those would oversubscribe the
# cgroup and slow the run down.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ENV_ASSIGNMENTS+=("OMP_NUM_THREADS=${OMP_NUM_THREADS}")

# HuggingFace tokenizers fork inside the dataloader workers and warn on every
# epoch otherwise.
export TOKENIZERS_PARALLELISM=false
ENV_ASSIGNMENTS+=("TOKENIZERS_PARALLELISM=false")

# Compute nodes here have no outbound network.  Left unset by default because the
# model is staged locally anyway; set OFFLINE=1 to make any accidental hub call
# fail fast instead of hanging on a connect timeout.
if [ "${OFFLINE:-0}" = "1" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    ENV_ASSIGNMENTS+=("HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1")
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ENV_ASSIGNMENTS+=("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
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
)

TRAIN_ARGS=(
    "++device=${DEVICE}"
    # nccl is the right backend on a single node of H100s; it is only consulted
    # when world_size > 1 (train_ds.py: use_ddp = world_size > 1).
    "++backend=nccl"
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
    # fp16 stays off even when bf16 is off: fp16 needs a GradScaler and brings
    # loss-scaling failure modes that bf16 does not, and this is a finetune of a
    # pretrained checkpoint where numerical drift is the thing to avoid.
    "++train_conf.use_fp16=false"
    "++train_conf.use_bf16=${BF16_EFFECTIVE}"
    "++train_conf.use_deepspeed=false"
)

if [ -n "${WARMUP_STEPS}" ]; then
    TRAIN_ARGS+=("++scheduler_conf.warmup_steps=${WARMUP_STEPS}")
fi

log_file="${OUTPUT_DIR}/train_${RUN_ID}.log"
# Keep hydra's own run directory beside the checkpoints instead of scattering
# outputs/<date>/<time>/ trees at whatever the container's cwd happens to be.
hydra_run_dir="${OUTPUT_DIR}/hydra/${RUN_ID}"

# --standalone picks its own free rendezvous port, which matters here: several
# jobs from the same user can land on the same node, and a fixed MASTER_PORT
# would make the second one fail to bind.
CMD=(
    "${PYTHON_BIN}"
    -m torch.distributed.run
    --standalone
    --nnodes=1
    "--nproc_per_node=${NPROC_PER_NODE}"
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

# Appended rather than written into the array above so that an invocation setting
# none of them produces a command that is byte-identical to the one this script
# emitted before these knobs existed.  Hydra overrides are order-independent, so
# trailing them is safe.
if [ -n "${INIT_PARAM}" ]; then
    CMD+=("++init_param=${INIT_PARAM}")
fi
if [ -n "${RICH_WEIGHT}" ]; then
    CMD+=("++model_conf.rich_loss_weight=${RICH_WEIGHT}")
fi
if [ -n "${EMO_MASK_TOKEN_ID}" ]; then
    CMD+=("++model_conf.emo_mask_token_id=${EMO_MASK_TOKEN_ID}")
fi

# ---------------------------------------------------------------------------
# OUTPUT_DIR concurrency lock
# ---------------------------------------------------------------------------
# Two runs sharing an OUTPUT_DIR is silent, total data loss.  Both write model.pt
# and model.pt.ep{N} to the same paths, and both run funasr's keep_nbest pruning,
# whose Trainer.save_checkpoint os.remove()s the checkpoints that lost the
# ranking -- including ones the other process is midway through writing.  Nothing
# in either log says anything is wrong; the damage only shows up as a corrupt
# torch.load hours later, after the compute is gone.
#
# This is not hypothetical here: scripts/submit_chunk_chain.sh submits several
# links against this same OUTPUT_DIR, and --dependency=afterany is the only thing
# keeping them apart.  A hand-submitted extra job, or a chain built with the
# wrong dependency type, puts two of them on the directory at once.
#
# Mechanism: an atomic mkdir plus an owner file inside it, as in
# finetune_chunk_mps.sh.  Two differences forced by the cluster:
#
#   1. The holder is identified by "hostname:pid", not a bare pid.  SLURM_JOB_ID
#      is unavailable (see the header) and jobs run on compute nodes while a
#      human inspecting the lock is on the login node, so a pid alone is
#      meaningless -- pid 4711 on t-gpu01 and pid 4711 on the login node are
#      unrelated processes, and treating one as evidence about the other would
#      either steal a live lock or refuse to break a dead one.
#   2. Liveness is a HEARTBEAT file the running job touches periodically, because
#      `ps -p` can only answer for the local host.  A heartbeat older than
#      LOCK_STALE_AFTER means the holder is gone (crashed, OOM-killed, or hit the
#      24 h wall clock, all of which happen on this cluster).  The threshold is
#      deliberately generous: a false "stale" verdict is the expensive mistake,
#      and a filesystem that is briefly slow under NFS load must not trigger one.
LOCK_DIR=""
LOCK_OWNER_FILE=""
LOCK_HEARTBEAT_FILE=""
LOCK_HELD=0
LOCK_HEARTBEAT_PID=""
LOCK_OWNER="$(hostname):$$"
# How often the running job refreshes the heartbeat.
LOCK_HEARTBEAT_INTERVAL="${LOCK_HEARTBEAT_INTERVAL:-60}"
# How stale it must be before another run may break the lock.  15 minutes is 15
# missed beats; nothing short of a dead job misses that many.
LOCK_STALE_AFTER="${LOCK_STALE_AFTER:-900}"

# mtime in epoch seconds.  GNU stat and BSD stat disagree on the flag and this
# script runs both inside a Linux container and on a developer's macOS box.
file_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null
}

# Seconds since the heartbeat was last touched, or empty if there is none.
lock_heartbeat_age() {
    local mtime now
    mtime="$(file_mtime "${LOCK_HEARTBEAT_FILE}")" || return 1
    [ -n "${mtime}" ] || return 1
    now="$(date +%s)"
    printf '%s' "$(( now - mtime ))"
}

lock_heartbeat_start() {
    # Backgrounded loop rather than a timer: no cron, no systemd, nothing to
    # install, and it dies with the shell.  It stops on its own if the lock
    # directory disappears, so a broken lock cannot leave a toucher running.
    (
        while [ -d "${LOCK_DIR}" ]; do
            touch "${LOCK_HEARTBEAT_FILE}" 2>/dev/null || exit 0
            sleep "${LOCK_HEARTBEAT_INTERVAL}"
        done
    ) &
    LOCK_HEARTBEAT_PID=$!
}

# Only ever release a lock THIS process owns.  If the owner file no longer names
# us, some other run took the directory over (our heartbeat stalled and we looked
# stale, say) and removing it would hand a third run a lock the second one is
# actively using -- the precise failure this whole section exists to prevent.
lock_release() {
    local rc=$? owner=""
    if [ -n "${LOCK_HEARTBEAT_PID}" ]; then
        kill "${LOCK_HEARTBEAT_PID}" 2>/dev/null || true
        LOCK_HEARTBEAT_PID=""
    fi
    if [ "${LOCK_HELD}" = "1" ]; then
        LOCK_HELD=0
        if [ -s "${LOCK_OWNER_FILE}" ]; then
            read -r owner <"${LOCK_OWNER_FILE}" 2>/dev/null || owner=""
        fi
        if [ "${owner}" = "${LOCK_OWNER}" ]; then
            rm -f "${LOCK_OWNER_FILE}" "${LOCK_HEARTBEAT_FILE}"
            rmdir "${LOCK_DIR}" 2>/dev/null || true
        else
            warn "lock ${LOCK_DIR} is now held by '${owner}', not by us (${LOCK_OWNER}); leaving it alone"
        fi
    fi
    return "${rc}"
}

# Is the holder definitely alive?  Only answerable when it is on THIS host, and
# even then not with `kill -0` alone: that fails both for "no such process"
# (ESRCH) and for "exists but you do not own it" (EPERM), and treating the second
# as a dead holder would make us steal a lock that is very much in use.
holder_is_local_and_alive() {
    local owner="$1" host pid
    host="${owner%%:*}"
    pid="${owner##*:}"
    [ "${host}" = "$(hostname)" ] || return 1
    case "${pid}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    ps -p "${pid}" -o pid= >/dev/null 2>&1 && return 0
    kill -0 "${pid}" 2>/dev/null
}

# First line of the owner file, or empty.  Retried: a competing run can be
# between its mkdir and its owner-file write, and treating that half-built lock
# as stale would let both processes think they hold it.
lock_read_owner() {
    local owner="" i
    for i in 1 2 3 4 5; do
        if [ -s "${LOCK_OWNER_FILE}" ]; then
            read -r owner <"${LOCK_OWNER_FILE}" 2>/dev/null || owner=""
            case "${owner}" in
                *:*) printf '%s' "${owner}"; return 0 ;;
            esac
        fi
        if [ "${i}" -lt 5 ]; then
            sleep 1
        fi
    done
    return 1
}

# Claim a stale lock by renaming it first.  rename(2) has exactly one winner, so
# if two runs spot the same dead lock simultaneously only one gets to clear it;
# the loser's mv fails, it loops, and it then contends on the normal mkdir.
# Without this both would rmdir-then-mkdir and both would believe they won.
lock_break() {
    local grave
    grave="${LOCK_DIR}.stale.$$.$(date +%s)"
    mv "${LOCK_DIR}" "${grave}" 2>/dev/null || return 1
    rm -f "${grave}/owner" "${grave}/heartbeat"
    # rmdir, not rm -rf: if anything unexpected is in there, fail loudly and
    # leave it for a human rather than deleting it.
    rmdir "${grave}" 2>/dev/null \
        || warn "stale lock remains at ${grave} (unexpected contents); remove it by hand"
    return 0
}

lock_acquire() {
    local dir="$1" attempt owner age
    LOCK_DIR="${dir}/.train.lock"
    LOCK_OWNER_FILE="${LOCK_DIR}/owner"
    LOCK_HEARTBEAT_FILE="${LOCK_DIR}/heartbeat"

    # Re-asserted before the first mkdir and a no-op until LOCK_HELD flips, so
    # the lock cannot outlive us via a signal delivered just after we take it.
    # On this cluster the signal that matters is the SIGTERM Slurm sends at the
    # wall clock: without this, every timed-out link would leave its lock behind
    # and the next link in the chain would refuse to start.  These are the same
    # handlers installed at startup (they release the lock and then write the
    # outcome sentinel); assigning them again here is deliberate belt-and-braces
    # and changes nothing about the lock itself.
    trap 'on_exit' EXIT
    trap 'on_signal INT 2' INT
    trap 'on_signal TERM 15' TERM

    for attempt in 1 2 3; do
        if mkdir "${LOCK_DIR}" 2>/dev/null; then
            LOCK_HELD=1
            printf '%s\n' "${LOCK_OWNER}" >"${LOCK_OWNER_FILE}"
            printf 'host=%s\npid=%s\nrun_id=%s\nstarted=%s\nscript=%s\noutput_dir=%s\n' \
                "$(hostname)" "$$" "${RUN_ID}" \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BASH_SOURCE[0]}" "${dir}" \
                >>"${LOCK_OWNER_FILE}"
            touch "${LOCK_HEARTBEAT_FILE}"
            lock_heartbeat_start
            info "lock:       ${LOCK_DIR} (owner ${LOCK_OWNER}, heartbeat every ${LOCK_HEARTBEAT_INTERVAL}s)"
            return 0
        fi

        owner="$(lock_read_owner)" || owner=""
        age="$(lock_heartbeat_age)" || age=""

        # Same host: ps is authoritative, so use it and skip the heartbeat.
        if [ -n "${owner}" ] && holder_is_local_and_alive "${owner}"; then
            die "$(printf '%s\n' \
                "another run already holds ${LOCK_DIR}" \
                "" \
                "  holder:      ${owner} (alive on this host)" \
                "  output_dir:  ${dir}" \
                "" \
                "Two runs sharing an output directory overwrite each other's model.pt and" \
                "delete each other's per-epoch checkpoints through funasr's keep_nbest" \
                "pruning, which destroys both runs silently.  Refusing to start." \
                "" \
                "Either wait for it to finish, or point this run somewhere else:" \
                "  OUTPUT_DIR=${dir}_2 sbatch ${BASH_SOURCE[0]}")"
        fi

        # Different host (the normal case): the heartbeat is the only evidence.
        if [ -n "${age}" ] && [ "${age}" -lt "${LOCK_STALE_AFTER}" ]; then
            die "$(printf '%s\n' \
                "another run already holds ${LOCK_DIR}" \
                "" \
                "  holder:      ${owner:-<unreadable>}" \
                "  heartbeat:   ${age}s ago (stale only after ${LOCK_STALE_AFTER}s)" \
                "  output_dir:  ${dir}" \
                "" \
                "The holder is on another node and is still refreshing its heartbeat, so it" \
                "is running.  Two runs sharing an output directory destroy each other's" \
                "checkpoints through funasr's keep_nbest pruning, silently.  Refusing to start." \
                "" \
                "If this job was submitted as part of a chain, the dependency is wrong:" \
                "links must be --dependency=afterany:<prev> so they never overlap." \
                "" \
                "Otherwise wait for the holder to finish, or:" \
                "  OUTPUT_DIR=${dir}_2 sbatch ${BASH_SOURCE[0]}")"
        fi

        if [ -z "${owner}" ]; then
            warn "lock ${LOCK_DIR} has no readable owner file; treating it as abandoned"
        elif [ -z "${age}" ]; then
            warn "lock ${LOCK_DIR} is held by ${owner} but has no heartbeat file (killed before it started, or written by an older version of this script); taking it over"
        else
            warn "lock ${LOCK_DIR} was held by ${owner}, whose heartbeat is ${age}s old (> ${LOCK_STALE_AFTER}s); the holder is gone (crash, OOM, or wall-clock timeout), taking it over"
        fi
        lock_break || true
    done

    die "could not acquire ${LOCK_DIR} after 3 attempts; another run is contending for it"
}

# Read-only view of the lock, for --dry-run.  Reports without creating anything.
lock_status() {
    local dir="$1" owner age
    LOCK_DIR="${dir}/.train.lock"
    LOCK_OWNER_FILE="${LOCK_DIR}/owner"
    LOCK_HEARTBEAT_FILE="${LOCK_DIR}/heartbeat"
    [ -d "${LOCK_DIR}" ] || return 0
    owner="$(lock_read_owner)" || owner=""
    age="$(lock_heartbeat_age)" || age=""
    if { [ -n "${owner}" ] && holder_is_local_and_alive "${owner}"; } \
        || { [ -n "${age}" ] && [ "${age}" -lt "${LOCK_STALE_AFTER}" ]; }; then
        warn "${dir} is currently locked by a live run (${owner:-unknown}, heartbeat ${age:-?}s ago); a real run would refuse to start until it finishes"
    else
        info "  note: a stale lock is present at ${LOCK_DIR} (owner ${owner:-unknown}, heartbeat ${age:-none}s ago); a real run would take it over"
    fi
}

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
print_command() {
    local i assignment
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
# which is what makes train_conf.resume=true usable across invocations and what
# makes the chained submission work at all.  That tolerance is exactly why the
# lock is needed: resume=true means a second run aimed here does not announce
# itself, it silently adopts and then corrupts the first run's checkpoints.  Take
# the lock before creating anything else, and hold it for the lifetime of the
# training process.
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
