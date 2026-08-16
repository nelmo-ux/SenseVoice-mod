#!/usr/bin/env bash
#
# Submit a chain of chunk-finetune jobs, each resuming where the last one stopped.
#
# Runs on the LOGIN NODE, not inside the container: it needs squeue/sbatch/sacct,
# and unlike finetune_chunk_slurm.sh it is free to reason about SLURM_* because
# the site sbatch wrapper only strips that environment on the way *into* the
# container.
#
# ---------------------------------------------------------------------------
# WHY A CHAIN
# ---------------------------------------------------------------------------
# The QOS MaxWall is 1-00:00:00 and the job asks for the full 24 h, which may not
# be enough for the whole run.  So instead of one long job, submit N jobs against
# the same OUTPUT_DIR, each depending on the previous one, and let
# train_conf.resume=true pick up from the last checkpoint.
#
# The dependency type is afterany, NOT afterok, and that is the whole point: a
# link that is killed at the wall clock exits non-zero, and afterok would leave
# the rest of the chain queued forever in DependencyNeverSatisfied -- precisely
# in the case the chain exists to handle.  afterany fires on any terminal state,
# so a timeout or a crash still hands off to the next link.
#
# Surplus links are cheap: RESUME=true means a link that finds an already
# finished run does its preflight, sees max_epoch already reached, and exits in
# under a minute.  So it is better to submit one link too many than one too few.
#
# afterany also serialises the chain, which is what keeps the OUTPUT_DIR lock in
# finetune_chunk_slurm.sh from ever firing: two links overlapping would be two
# runs pruning each other's checkpoints.
#
# ---------------------------------------------------------------------------
# WHY THE SUBMIT COUNT IS CHECKED FIRST
# ---------------------------------------------------------------------------
# MaxSubmitJobsPU=4 is a per-user cap on jobs *in the queue*, counting pending
# ones -- and a chain is pending by construction.  Over the cap, sbatch rejects
# the submission outright, which would leave a half-built chain: some links
# submitted with dependencies, the rest missing, and the tail of the chain
# silently never running.  Counting first and refusing as a whole is the only way
# to keep that from happening.
#
# The reserve exists for the same reason from the other direction: filling the
# queue to exactly 4 means the pseudo-labeling job cannot be submitted at all
# until this chain drains, which can be a day or more.
#
# Usage:
#   scripts/submit_chunk_chain.sh                       # 2 links, default script
#   scripts/submit_chunk_chain.sh -n 4                  # 4 links
#   scripts/submit_chunk_chain.sh -n 3 --dry-run        # print, do not submit
#   scripts/submit_chunk_chain.sh -n 2 --export=MAX_EPOCH=8   # extra sbatch args
#
# Anything this script does not recognise, and everything after --, is forwarded
# to sbatch unchanged.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# ---------------------------------------------------------------------------
# Site limits (Slurm 23.02.8, partition `research`, per-user aggregate)
# ---------------------------------------------------------------------------
# MaxSubmitJobsPU: jobs a single user may have in the queue at once, pending
# included.  Overridable only because a QOS change should not require editing
# this file in a hurry.
MAX_SUBMIT_JOBS_PU="${MAX_SUBMIT_JOBS_PU:-4}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LINKS=2
JOBSCRIPT="${repo_root}/finetune_chunk_slurm.sh"
# Keep one submit slot free for the concurrent pseudo-labeling job.
RESERVE=1
DRY_RUN=0

# Overridable so the whole script can be exercised without a scheduler.
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"

info() { printf '%s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Submit a chain of chunk-finetune jobs, each resuming from the last checkpoint.

Usage:
  submit_chunk_chain.sh [-n LINKS] [-j JOBSCRIPT] [-r RESERVE] [--dry-run]
                        [extra sbatch args...]

Options:
  -n, --links N      Number of chained jobs (default: 2).
  -j, --jobscript P  Job script to submit (default: <repo>/finetune_chunk_slurm.sh).
  -r, --reserve N    Submit slots to leave free for other jobs (default: 1).
                     The QOS allows 4 queued jobs per user; this keeps room for
                     the concurrent pseudo-labeling job.
      --dry-run      Print the sbatch commands without submitting anything.
  -h, --help         This message.

Everything unrecognised, and everything after --, is forwarded to sbatch.
Each link after the first is submitted with --dependency=afterany:<previous>,
so a link that times out or crashes still hands off to the next one.

Environment:
  MAX_SUBMIT_JOBS_PU  QOS submit cap to respect (default: 4).
  SBATCH_BIN/SQUEUE_BIN  Override the scheduler binaries (for testing).
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
# The first unrecognised argument ends option parsing: everything from there on
# belongs to sbatch, and sbatch options start with '-' too, so there is no way to
# keep looking for ours without risking swallowing one of theirs.
SBATCH_EXTRA=()

require_value() {
    [ "$2" -ge 2 ] || die "$1 requires a value"
}

is_positive_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        0) return 1 ;;
        *) return 0 ;;
    esac
}

is_non_negative_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--links)
            require_value "$1" "$#"
            LINKS="$2"
            shift 2
            ;;
        --links=*)
            LINKS="${1#*=}"
            shift
            ;;
        -j|--jobscript)
            require_value "$1" "$#"
            JOBSCRIPT="$2"
            shift 2
            ;;
        --jobscript=*)
            JOBSCRIPT="${1#*=}"
            shift
            ;;
        -r|--reserve)
            require_value "$1" "$#"
            RESERVE="$2"
            shift 2
            ;;
        --reserve=*)
            RESERVE="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            SBATCH_EXTRA+=("$@")
            break
            ;;
        *)
            SBATCH_EXTRA+=("$@")
            break
            ;;
    esac
done

is_positive_int "${LINKS}" || die "--links must be a positive integer, got '${LINKS}'"
is_non_negative_int "${RESERVE}" || die "--reserve must be a non-negative integer, got '${RESERVE}'"
is_positive_int "${MAX_SUBMIT_JOBS_PU}" || die "MAX_SUBMIT_JOBS_PU must be a positive integer, got '${MAX_SUBMIT_JOBS_PU}'"

[ -f "${JOBSCRIPT}" ] || die "job script not found: ${JOBSCRIPT}"
[ -r "${JOBSCRIPT}" ] || die "job script is not readable: ${JOBSCRIPT}"

# ---------------------------------------------------------------------------
# Submit-cap check
# ---------------------------------------------------------------------------
# -h drops the header, so the line count is the job count.  Job ARRAY tasks would
# each be a line here; that is the conservative direction (it can only make us
# refuse a submission that would have fit), and no job here uses arrays.
count_queued_jobs() {
    local user="${USER:-$(id -un)}" out n
    command -v "${SQUEUE_BIN}" >/dev/null 2>&1 || return 1
    out="$("${SQUEUE_BIN}" -h -u "${user}" -o '%i' 2>/dev/null)" || return 1
    # grep -c exits 1 when it counts zero, which set -e would treat as a failure
    # of the whole check rather than the "queue is empty" it actually means.
    n="$(printf '%s\n' "${out}" | grep -c '[^[:space:]]' || true)"
    printf '%s' "${n:-0}"
}

allowed=$(( MAX_SUBMIT_JOBS_PU - RESERVE ))
if [ "${allowed}" -lt 1 ]; then
    die "--reserve ${RESERVE} leaves no submit slots at all (MaxSubmitJobsPU=${MAX_SUBMIT_JOBS_PU})"
fi

if current="$(count_queued_jobs)"; then
    info "queue: ${current} job(s) already submitted as ${USER:-$(id -un)}; cap ${MAX_SUBMIT_JOBS_PU}, reserving ${RESERVE} -> ${allowed} usable"
    if [ "$(( current + LINKS ))" -gt "${allowed}" ]; then
        cat >&2 <<EOF
Error: submitting ${LINKS} more job(s) would exceed the submit budget.

  already queued:      ${current}
  requested links:     ${LINKS}
  MaxSubmitJobsPU:     ${MAX_SUBMIT_JOBS_PU}
  reserved (--reserve): ${RESERVE}
  usable slots:        ${allowed}

MaxSubmitJobsPU counts PENDING jobs, and a chain is pending by construction, so
sbatch would reject the submission partway through and leave a broken chain: the
first links queued with dependencies and the rest missing.

Options:
  * wait for running jobs to finish        (${SQUEUE_BIN} -u ${USER:-$(id -un)})
  * submit fewer links now and top up later -- surplus links are cheap, missing
    ones just mean the run resumes when you next submit
  * lower the reserve with -r 0, but then the pseudo-labeling job cannot be
    submitted until this chain drains
EOF
        exit 1
    fi
else
    if [ "${DRY_RUN}" = "1" ]; then
        warn "${SQUEUE_BIN} unavailable; skipping the MaxSubmitJobsPU=${MAX_SUBMIT_JOBS_PU} check (dry run)"
    else
        die "${SQUEUE_BIN} unavailable or failed; refusing to submit blind against MaxSubmitJobsPU=${MAX_SUBMIT_JOBS_PU}. Run this on a login node, or set SQUEUE_BIN."
    fi
fi

if [ "${DRY_RUN}" != "1" ]; then
    command -v "${SBATCH_BIN}" >/dev/null 2>&1 \
        || die "${SBATCH_BIN} not found; this script runs on the login node, not inside the container"
fi

# ---------------------------------------------------------------------------
# Job id extraction
# ---------------------------------------------------------------------------
# The site sbatch wrapper is a Python program that docker-pulls the --container
# image before forwarding to the real sbatch, and it prints its own lines first:
#
#     ContainerImage: <cluster-registry>/<project>/sensevoice-chunk:...
#     Submitted batch job 12345
#
# So neither "last line" nor "last field" is safe -- the wrapper is free to print
# more after the id, and an image tag ending in digits would satisfy a naive
# field grab.  Match the sbatch sentence itself and take the trailing number.
# --parsable is deliberately not used: it would suppress that sentence, but the
# wrapper's own output is not covered by it, so the parsing problem would remain
# while the format became one this comment no longer describes.
parse_job_id() {
    printf '%s\n' "$1" \
        | sed -n 's/^.*Submitted batch job \([0-9][0-9]*\).*$/\1/p' \
        | tail -n 1
}

# ---------------------------------------------------------------------------
# Submit the chain
# ---------------------------------------------------------------------------
job_ids=()
job_deps=()
prev_id=""

info ""
info "chain: ${LINKS} link(s) of ${JOBSCRIPT}"
if [ "${#SBATCH_EXTRA[@]}" -gt 0 ]; then
    info "extra sbatch args: ${SBATCH_EXTRA[*]}"
fi
info ""

for (( link = 1; link <= LINKS; link++ )); do
    cmd=("${SBATCH_BIN}")
    if [ -n "${prev_id}" ]; then
        cmd+=("--dependency=afterany:${prev_id}")
    fi
    if [ "${#SBATCH_EXTRA[@]}" -gt 0 ]; then
        cmd+=("${SBATCH_EXTRA[@]}")
    fi
    cmd+=("${JOBSCRIPT}")

    if [ "${DRY_RUN}" = "1" ]; then
        # The dependency of a link is only known once the previous one is
        # submitted, so a dry run has to show a placeholder rather than pretend.
        if [ "${link}" -eq 1 ]; then
            printf 'link %d: %s\n' "${link}" "$(printf '%q ' "${cmd[@]}")"
        else
            printf 'link %d: %s\n' "${link}" \
                "$(printf '%q ' "${cmd[@]}" | sed "s/afterany:${prev_id}/afterany:<jobid-of-link-$(( link - 1 ))>/")"
        fi
        prev_id="PLACEHOLDER_$(( link - 1 ))"
        job_ids+=("<link-${link}>")
        job_deps+=("${prev_id}")
        continue
    fi

    info "link ${link}: ${cmd[*]}"
    if ! out="$("${cmd[@]}" 2>&1)"; then
        printf '%s\n' "${out}" >&2
        # A partial chain is a real state, not a rollback opportunity: the links
        # that did go in are already queued and will run.  Say which they are.
        if [ "${#job_ids[@]}" -gt 0 ]; then
            die "sbatch failed on link ${link}. Links 1..$(( link - 1 )) ARE submitted (${job_ids[*]}); either let them run or cancel them with: scancel ${job_ids[*]}"
        fi
        die "sbatch failed on link ${link}; nothing was submitted."
    fi
    printf '%s\n' "${out}"

    job_id="$(parse_job_id "${out}")"
    if [ -z "${job_id}" ]; then
        # sbatch exited 0, so link ${link} is probably QUEUED -- we just cannot
        # name it, which means we cannot make the next link depend on it.
        # Stopping here leaves an unchained job running rather than a silent
        # overlap with whatever gets submitted next.
        die "$(printf '%s\n' \
            "could not find 'Submitted batch job <id>' in the wrapper output for link ${link}, but sbatch reported success." \
            "" \
            "Link ${link} is probably queued and cannot be chained to. Check and clean up before resubmitting:" \
            "  ${SQUEUE_BIN} -u ${USER:-$(id -un)}" \
            "${job_ids[*]:+  already submitted by this run: ${job_ids[*]}}")"
    fi

    job_ids+=("${job_id}")
    job_deps+=("${prev_id}")
    prev_id="${job_id}"
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
info ""
if [ "${DRY_RUN}" = "1" ]; then
    info "dry run: nothing was submitted."
    exit 0
fi

info "chain submitted:"
for (( i = 0; i < ${#job_ids[@]}; i++ )); do
    if [ -z "${job_deps[$i]}" ]; then
        info "  link $(( i + 1 )): job ${job_ids[$i]}  (no dependency -- starts when resources free up)"
    else
        info "  link $(( i + 1 )): job ${job_ids[$i]}  (afterany:${job_deps[$i]})"
    fi
done

info ""
info "watch it:"
info "  ${SQUEUE_BIN} -u ${USER:-$(id -un)} -o '%.10i %.10P %.12j %.8T %.10M %.6D %R'"
# --output=%x-%j.out in the job script, resolved relative to the submit dir.
info "  tail -f sv-chunk-${job_ids[0]}.out        # link 1 (the running one)"
if [ "${#job_ids[@]}" -gt 1 ]; then
    info "  ls sv-chunk-{$(IFS=,; printf '%s' "${job_ids[*]}")}.out   # every link"
fi
info "  sacct -j $(IFS=,; printf '%s' "${job_ids[*]}") --format=JobID,JobName,State,Elapsed,ExitCode,Start,End"
info ""
info "cancel the whole chain:"
info "  scancel ${job_ids[*]}"
