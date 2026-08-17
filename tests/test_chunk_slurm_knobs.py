"""Tests for the INIT_PARAM / RICH_WEIGHT / EMO_MASK_TOKEN_ID knobs and the
constant-``emo_target`` preflight check in ``finetune_chunk_slurm.sh``.

The script is driven end to end as a subprocess -- there is nothing to import --
using its two non-destructive entry points:

``--dry-run``
    Runs every check, downgrades failures to warnings, prints the fully resolved
    torchrun command and exits without taking the OUTPUT_DIR lock or launching
    training.  Most assertions here read that resolved command.

a real (non-dry) run that is *designed to fail in preflight*
    The only way to observe the outcome contract -- ``${OUTPUT_DIR}/.job_status``
    plus the ``SENSEVOICE_JOB_FAILED`` log marker -- because ``--dry-run``
    deliberately writes neither.  Preflight aborts before the lock is taken and
    long before torchrun is executed, so no training is ever started.

WHAT THESE TESTS CANNOT COVER, AND WHY
Nothing here proves the emitted overrides do the right thing *inside funasr*:
that needs a GPU, the real corpus and the container.  What is pinned locally is
the contract this script owns -- which flags are emitted, which values are
refused before a GPU allocation is burnt, and that a failure is still reported
through the sentinel.  See the module docstring of the cluster runbook for the
checks that must be repeated on the cluster.
"""

import json
import os
import shutil
import subprocess
import textwrap

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "finetune_chunk_slurm.sh")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "SenseVoiceSmall")

# The three overrides under test, in the form they appear on the command line.
INIT_PARAM_FLAG = "++init_param="
RICH_FLAG = "++model_conf.rich_loss_weight="
EMO_FLAG = "++model_conf.emo_mask_token_id="
ALL_FLAGS = (INIT_PARAM_FLAG, RICH_FLAG, EMO_FLAG)

# The id of the single token <|SER|> in the model's bpe model.  Hardcoded rather
# than looked up so the test still states the intent when the model is absent.
SER_TOKEN_ID = "24991"
# The same thing on the manifest side: emo_target holds the token, not the id.
EMO_SENTINEL = "<|SER|>"


pytestmark = pytest.mark.skipif(
    not os.path.isfile(SCRIPT), reason="finetune_chunk_slurm.sh not present"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gpu_stub(tmp_path_factory):
    """A stub ``nvidia-smi`` on PATH.

    The script check_fail()s when nvidia-smi is missing, which on a laptop aborts
    a real run *before* preflight is ever reached -- so the preflight-failure
    tests below could not run at all.  Stubbing it also makes the tests behave
    identically on a machine that happens to have a real GPU.
    """
    bin_dir = tmp_path_factory.mktemp("stub_bin")
    stub = bin_dir / "nvidia-smi"
    stub.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            case "$*" in
                *-L*) printf 'GPU 0: NVIDIA H100 (UUID: GPU-stub)\\n' ;;
                *--query-gpu=*) printf '0, NVIDIA H100 80GB HBM3, 81559 MiB, 550.54.15\\n' ;;
            esac
            """
        )
    )
    stub.chmod(0o755)
    return str(bin_dir)


@pytest.fixture(scope="module")
def site_script(tmp_path_factory):
    """A copy of the script with the site-specific placeholders filled in.

    The committed script carries ``<cluster-registry>`` / ``<project>`` /
    ``<user>`` placeholders on purpose (this repo is public), and the very first
    check refuses to run while a placeholder is present.  ``--dry-run`` downgrades
    that to a warning, but a real run exits on it, so the non-dry-run tests need
    an "edited" copy.  It is written outside the repo so the checkout stays clean.
    """
    src = open(SCRIPT, encoding="utf-8").read()
    edited = (
        src.replace("<cluster-registry>", "reg.example.invalid")
        .replace("<project>", "proj")
        .replace("<user>", "someuser")
    )
    dest = tmp_path_factory.mktemp("site") / "finetune_chunk_slurm.sh"
    dest.write_text(edited, encoding="utf-8")
    dest.chmod(0o755)
    return str(dest)


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    """Three manifests differing only in their ``emo_target`` field.

    ``source`` deliberately points at files that do not exist: the audio-presence
    check is a separate preflight failure and is irrelevant here, and generating
    real audio would make the suite slow for no gain.  The emo_target check runs
    before it and does not depend on it.
    """
    d = tmp_path_factory.mktemp("manifests")

    def write(name, records):
        path = d / name
        with open(path, "w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec) + "\n")
        return str(path)

    def record(i, **extra):
        return dict(source=f"/nonexistent/{i}.wav", target="xin chao", source_len=500, **extra)

    return {
        # The round-1/2 defect: every clip stamped with the same label.
        "const": write("const.jsonl", [record(i, emo_target="<|NEUTRAL|>") for i in range(10)]),
        "varied": write(
            "varied.jsonl",
            [
                record(i, emo_target="<|NEUTRAL|>" if i % 2 else "<|HAPPY|>")
                for i in range(10)
            ],
        ),
        # An older manifest that predates the field entirely.
        "absent": write("absent.jsonl", [record(i) for i in range(10)]),
        # What prepare_vn_data.py --emo-labels (or make_smoke_data.py --emo-mix)
        # produces: real emotion tokens with a share of the mask sentinel.
        "sentinel": write(
            "sentinel.jsonl",
            [
                record(i, emo_target=EMO_SENTINEL if i < 3 else ("<|HAPPY|>" if i % 2 else "<|SAD|>"))
                for i in range(10)
            ],
        ),
    }


def run_script(script, gpu_stub, env=None, dry_run=True, cwd=None):
    """Invoke the script, returning the CompletedProcess with merged output."""
    full_env = dict(os.environ)
    full_env["PATH"] = gpu_stub + os.pathsep + full_env["PATH"]
    # The script resolves the repo from its own location; the site copy lives
    # outside the checkout, so it has to be told.
    full_env["WORKSPACE"] = REPO_ROOT
    # cuda is unavailable locally; DEVICE=cpu keeps that from adding noise.
    full_env["DEVICE"] = "cpu"
    # Never inherit these from the developer's shell.
    for key in (
        "INIT_PARAM",
        "RICH_WEIGHT",
        "EMO_MASK_TOKEN_ID",
        "SMOKE",
        "RESUME",
        "TIME_CEILING_HOURS",
    ):
        full_env.pop(key, None)
    full_env.update(env or {})

    cmd = [script] + (["--dry-run"] if dry_run else [])
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )


def resolved_command(output):
    """Extract the resolved torchrun command from --dry-run output."""
    marker = "resolved command"
    assert marker in output, f"no resolved command in output:\n{output}"
    tail = output.split(marker, 1)[1]
    # The command block ends at the blank line before the dry-run summary.
    end = tail.find("\ndry run finished")
    assert end != -1, f"no dry-run summary in output:\n{output}"
    return tail[:end]


def command_flag_values(command, flag):
    """Every value given to ``flag`` in the resolved command, in order."""
    values = []
    for token in command.replace("\\\n", " ").split():
        if token.startswith(flag):
            values.append(token[len(flag):])
    return values


# ---------------------------------------------------------------------------
# Default behaviour: the knobs must be invisible when unset
# ---------------------------------------------------------------------------
def test_no_knobs_set_emits_none_of_the_overrides(gpu_stub):
    """Round-2 reproducibility: unset knobs must add nothing to the command."""
    result = run_script(SCRIPT, gpu_stub)
    command = resolved_command(result.stdout)
    for flag in ALL_FLAGS:
        assert flag not in command, f"{flag} leaked into the default command"


def test_knobs_are_purely_additive(gpu_stub):
    """The command with all three set is the default command plus three lines.

    Stronger than "the flags are absent by default": it pins that enabling them
    does not reorder, drop or alter any pre-existing argument.  Expressed as a
    set difference rather than against a hardcoded command so that unrelated
    future changes to the script do not make this test lie.
    """
    base = resolved_command(run_script(SCRIPT, gpu_stub).stdout)
    withal = resolved_command(
        run_script(
            SCRIPT,
            gpu_stub,
            env={
                "INIT_PARAM": os.path.join(MODEL_DIR, "model.pt"),
                "RICH_WEIGHT": "0.3",
                "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
            },
        ).stdout
    )

    def args(block):
        # Drop the run-id, which changes between invocations by design.
        return [
            line.strip().rstrip(" \\")
            for line in block.splitlines()
            if line.strip() and "hydra.run.dir" not in line and "tee " not in line
        ]

    added = [a for a in args(withal) if a not in args(base)]
    assert [a.split("=", 1)[0] for a in added] == [
        "++init_param",
        "++model_conf.rich_loss_weight",
        "++model_conf.emo_mask_token_id",
    ]
    assert [a for a in args(base) if a not in args(withal)] == []


# ---------------------------------------------------------------------------
# Each knob appends exactly its own override, exactly once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "var,value,flag",
    [
        ("RICH_WEIGHT", "0.3", RICH_FLAG),
        ("EMO_MASK_TOKEN_ID", SER_TOKEN_ID, EMO_FLAG),
    ],
)
def test_knob_appends_its_override_once(gpu_stub, var, value, flag):
    result = run_script(SCRIPT, gpu_stub, env={var: value})
    command = resolved_command(result.stdout)

    assert command_flag_values(command, flag) == [value]
    # and does not drag the other two along with it
    for other in ALL_FLAGS:
        if other != flag:
            assert other not in command


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "model.pt")),
    reason="no local checkpoint to point INIT_PARAM at",
)
def test_init_param_appends_its_override_once(gpu_stub):
    ckpt = os.path.join(MODEL_DIR, "model.pt")
    result = run_script(SCRIPT, gpu_stub, env={"INIT_PARAM": ckpt})
    command = resolved_command(result.stdout)

    assert command_flag_values(command, INIT_PARAM_FLAG) == [ckpt]
    assert RICH_FLAG not in command
    assert EMO_FLAG not in command


# ---------------------------------------------------------------------------
# Invalid values are refused in preflight, by name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "var,value",
    [
        ("RICH_WEIGHT", "abc"),
        ("RICH_WEIGHT", "-1"),
        ("RICH_WEIGHT", ""),  # whitespace-only is treated as unset, see below
        ("EMO_MASK_TOKEN_ID", "-5"),
        ("EMO_MASK_TOKEN_ID", "24991.5"),
        ("EMO_MASK_TOKEN_ID", "SER"),
        ("INIT_PARAM", "/nonexistent/round2/model.pt.ep3"),
    ],
)
def test_invalid_value_fails_preflight_naming_the_variable(gpu_stub, var, value):
    result = run_script(SCRIPT, gpu_stub, env={var: value})
    fails = [ln for ln in result.stdout.splitlines() if "FAIL:" in ln]

    if value == "":
        # Empty means "unset", which is the documented default -- not an error.
        assert not any(var in ln for ln in fails)
        return

    assert any(var in ln for ln in fails), (
        f"no preflight FAIL naming {var} for value {value!r}; got:\n"
        + "\n".join(fails)
    )


def test_preflight_failure_still_writes_job_status_and_marker(
    site_script, gpu_stub, manifests, tmp_path
):
    """The outcome contract must survive a failure introduced by the new knobs.

    The scheduler's own exit code is worthless here (a job whose whole body is
    ``exit 42`` is recorded COMPLETED 0:0), so ``.job_status`` and the log marker
    are the only trustworthy signal that the run refused to start.
    """
    out = tmp_path / "out"
    result = run_script(
        site_script,
        gpu_stub,
        dry_run=False,
        env={
            "MODEL_DIR": MODEL_DIR,
            "TRAIN_JSONL": manifests["varied"],
            "VAL_JSONL": manifests["varied"],
            "OUTPUT_DIR": str(out),
            "RICH_WEIGHT": "abc",
        },
    )

    assert result.returncode == 1
    assert any(
        "FAIL:" in ln and "RICH_WEIGHT" in ln for ln in result.stdout.splitlines()
    ), result.stdout

    # The marker: single line, fixed leading token, greppable from outside.
    assert "SENSEVOICE_JOB_FAILED rc=1" in result.stdout
    assert "SENSEVOICE_JOB_OK" not in result.stdout
    assert "preflight=failed" in result.stdout

    status = (out / ".job_status").read_text(encoding="utf-8")
    assert "result=FAILURE" in status
    assert "exit_code=1" in status
    assert "preflight=failed" in status
    assert "marker=SENSEVOICE_JOB_FAILED" in status

    # Preflight aborts before the lock is taken and before torchrun runs.
    assert not (out / "model.pt").exists()


# ---------------------------------------------------------------------------
# The constant-emo_target check
# ---------------------------------------------------------------------------
def _emo_lines(output):
    return [ln for ln in output.splitlines() if "emo_target is the single value" in ln]


def test_constant_emo_target_is_reported(gpu_stub, manifests):
    """The round-1/2 defect: every clip stamped <|NEUTRAL|>."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["const"], "VAL_JSONL": manifests["varied"]},
    )
    lines = _emo_lines(result.stdout)

    assert len(lines) == 1, f"expected exactly one report, got {lines}"
    line = lines[0]
    assert line.strip().startswith("note:"), "should be a note when masking is not requested"
    assert "train:" in line
    assert "<|NEUTRAL|>" in line, "the offending value must be named"
    assert "10 records" in line, "the record count must be reported"


def test_varied_emo_target_is_silent(gpu_stub, manifests):
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["varied"], "VAL_JSONL": manifests["varied"]},
    )
    assert _emo_lines(result.stdout) == []


def test_missing_emo_target_field_is_silent(gpu_stub, manifests):
    """Older manifests have no emo_target at all; that is not the defect."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["absent"], "VAL_JSONL": manifests["absent"]},
    )
    assert _emo_lines(result.stdout) == []


def test_constant_emo_target_is_fatal_when_masking_requested(gpu_stub, manifests):
    """Masking the emotion slot against constant labels is a contradiction."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "TRAIN_JSONL": manifests["const"],
            "VAL_JSONL": manifests["varied"],
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    lines = _emo_lines(result.stdout)

    assert len(lines) == 1
    assert "FAIL:" in lines[0], f"expected a hard failure, got: {lines[0]}"


def test_constant_emo_target_is_only_a_note_under_smoke(gpu_stub, manifests, tmp_path):
    """SMOKE is exempt: a DEFAULT smoke corpus is constant by construction.

    scripts/make_smoke_data.py gives every generated clip the same emo_target
    unless it is asked for a mixture, so failing here would block the one run that
    exists to prove these overrides load at all.  The exemption covers the default
    smoke corpus only -- it is not a claim that smoke manifests never vary, and it
    deliberately does NOT extend to the sentinel check (see the test below).
    """
    smoke_out = tmp_path / "smoke"
    smoke_data = smoke_out / "smoke_data"
    smoke_data.mkdir(parents=True)
    # SMOKE overrides TRAIN_JSONL/VAL_JSONL to fixed names under OUTPUT_DIR.
    for name in ("smoke_train.jsonl", "smoke_val.jsonl"):
        shutil.copyfile(manifests["const"], smoke_data / name)

    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "SMOKE": "1",
            "OUTPUT_DIR_SMOKE": str(smoke_out),
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    lines = _emo_lines(result.stdout)

    assert lines, "the constant label should still be reported under SMOKE"
    assert all("FAIL:" not in ln for ln in lines), f"SMOKE must not fail on it: {lines}"


# ---------------------------------------------------------------------------
# The <|SER|> sentinel / EMO_MASK_TOKEN_ID cross-check
# ---------------------------------------------------------------------------
def _sentinel_lines(output, kind):
    needle = "carry the <|SER|> emotion sentinel"
    if kind == "warn":
        needle = "is set but not one of"
    return [ln for ln in output.splitlines() if needle in ln]


def test_sentinel_without_mask_is_fatal(gpu_stub, manifests):
    """model.py raises on this at step 1; preflight catches it before the queue."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["sentinel"], "VAL_JSONL": manifests["varied"]},
    )
    lines = _sentinel_lines(result.stdout, "fatal")

    assert len(lines) == 1, f"expected one report, got {lines}"
    assert "FAIL:" in lines[0]
    assert "3 record(s)" in lines[0], "the sentinel count must be reported"
    # Both halves of the fix must be named -- which is right depends on intent.
    assert "EMO_MASK_TOKEN_ID=24991" in lines[0]
    assert "--emo-labels" in lines[0]


def test_sentinel_in_val_manifest_is_also_fatal(gpu_stub, manifests):
    """Trainer.validate_epoch calls the same forward_step, so val raises too."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["varied"], "VAL_JSONL": manifests["sentinel"]},
    )
    lines = _sentinel_lines(result.stdout, "fatal")

    assert len(lines) == 1
    assert "FAIL:" in lines[0]
    assert "val:" in lines[0]


def test_sentinel_with_mask_set_is_clean(gpu_stub, manifests):
    """The intended round-3 configuration must pass without complaint."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "TRAIN_JSONL": manifests["sentinel"],
            "VAL_JSONL": manifests["sentinel"],
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    assert _sentinel_lines(result.stdout, "fatal") == []
    assert _sentinel_lines(result.stdout, "warn") == []


def test_mask_set_without_any_sentinel_warns_but_does_not_fail(gpu_stub, manifests):
    """Most likely the round-2 corpus; legitimate for an ablation, so not fatal."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "TRAIN_JSONL": manifests["varied"],
            "VAL_JSONL": manifests["varied"],
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    warns = _sentinel_lines(result.stdout, "warn")

    assert warns, "the mismatch must be said out loud"
    assert all("FAIL:" not in ln for ln in warns), "must not be fatal"
    assert any("note:" in ln for ln in warns)
    assert any("--emo-labels" in ln for ln in warns)


def test_sentinel_check_silent_without_emo_target_field(gpu_stub, manifests):
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"TRAIN_JSONL": manifests["absent"], "VAL_JSONL": manifests["absent"]},
    )
    assert _sentinel_lines(result.stdout, "fatal") == []
    assert _sentinel_lines(result.stdout, "warn") == []


def test_sentinel_without_mask_is_fatal_even_under_smoke(gpu_stub, manifests, tmp_path):
    """A --emo-mix smoke run that forgets EMO_MASK_TOKEN_ID proves nothing.

    Deliberately NOT exempt, unlike the constant-label check: the forward pass
    raises just as reliably on generated data, so exempting it would only move the
    same crash a few minutes later while letting the run look like it had
    exercised the masking path.
    """
    smoke_out = tmp_path / "smoke"
    smoke_data = smoke_out / "smoke_data"
    smoke_data.mkdir(parents=True)
    for name in ("smoke_train.jsonl", "smoke_val.jsonl"):
        shutil.copyfile(manifests["sentinel"], smoke_data / name)

    result = run_script(
        SCRIPT,
        gpu_stub,
        env={"SMOKE": "1", "OUTPUT_DIR_SMOKE": str(smoke_out)},
    )
    lines = _sentinel_lines(result.stdout, "fatal")

    assert lines, "SMOKE must not exempt this check"
    assert any("FAIL:" in ln for ln in lines), f"must stay fatal under SMOKE: {lines}"


def test_sentinel_failure_writes_job_status_and_marker(
    site_script, gpu_stub, manifests, tmp_path
):
    """The sentinel refusal must reach the outcome sentinel like any other."""
    out = tmp_path / "out"
    result = run_script(
        site_script,
        gpu_stub,
        dry_run=False,
        env={
            "MODEL_DIR": MODEL_DIR,
            "TRAIN_JSONL": manifests["sentinel"],
            "VAL_JSONL": manifests["varied"],
            "OUTPUT_DIR": str(out),
        },
    )

    assert result.returncode == 1
    assert any(
        "FAIL:" in ln and "<|SER|> emotion sentinel" in ln
        for ln in result.stdout.splitlines()
    ), result.stdout
    assert "SENSEVOICE_JOB_FAILED rc=1" in result.stdout
    assert "SENSEVOICE_JOB_OK" not in result.stdout

    status = (out / ".job_status").read_text(encoding="utf-8")
    assert "result=FAILURE" in status
    assert "preflight=failed" in status
    assert "marker=SENSEVOICE_JOB_FAILED" in status


def test_emo_target_distribution_is_reported_exactly(gpu_stub, manifests):
    """The distribution line is counted over every record, not a sample."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "TRAIN_JSONL": manifests["sentinel"],
            "VAL_JSONL": manifests["sentinel"],
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    lines = [ln for ln in result.stdout.splitlines() if "distinct value(s) over" in ln]

    assert len(lines) == 2, f"one line per manifest, got {lines}"
    train = next(ln for ln in lines if "train:" in ln)
    # 3x <|SER|>, 4x <|HAPPY|> (odd i), 3x <|SAD|> over 10 records.
    assert "3 distinct value(s) over 10/10 records" in train
    assert "<|SER|> sentinel 3 (30.0%)" in train


# ---------------------------------------------------------------------------
# Wall-clock ceiling
# ---------------------------------------------------------------------------
def test_requested_walltime_over_ceiling_warns(gpu_stub):
    """The committed header asks for 24 h; sacct shows 8 h being granted.

    A warning rather than a failure: the ceiling is an observation of two jobs
    rather than a read of enforced configuration, and the --time line cannot be
    templated from the environment, so a fatal check would block every run behind
    a manual edit of a site-specific line.
    """
    result = run_script(SCRIPT, gpu_stub, env={"TIME_CEILING_HOURS": "8"})
    warns = [ln for ln in result.stdout.splitlines() if "WALL CLOCK:" in ln]

    assert len(warns) == 1, f"expected one wall-clock warning, got {warns}"
    line = warns[0]
    assert "note:" in line, "must be a note, not a fatal error"
    assert "--time=24:00:00" in line, "the request must be named"
    assert "8 h" in line, "the believed ceiling must be named"
    # The consequence and the mitigation both have to be in the message.
    assert "exit code is discarded" in line
    assert "submit_chunk_chain.sh" in line
    # And it must be honest about what it did not check.
    assert "cannot be read from inside the container" in line

    assert not any(
        "FAIL:" in ln and "WALL CLOCK" in ln for ln in result.stdout.splitlines()
    )


def test_requested_walltime_within_ceiling_is_confirmed_not_warned(gpu_stub):
    """Raising the ceiling to a confirmed value silences the warning."""
    result = run_script(SCRIPT, gpu_stub, env={"TIME_CEILING_HOURS": "24"})

    assert not [ln for ln in result.stdout.splitlines() if "WALL CLOCK:" in ln]
    confirmations = [
        ln for ln in result.stdout.splitlines() if ln.strip().startswith("wall clock:")
    ]
    assert len(confirmations) == 1, confirmations
    assert "within the believed 24 h ceiling" in confirmations[0]


@pytest.mark.parametrize("value", ["abc", "0", "-4"])
def test_invalid_time_ceiling_fails_preflight_by_name(gpu_stub, value):
    result = run_script(SCRIPT, gpu_stub, env={"TIME_CEILING_HOURS": value})
    assert any(
        "FAIL:" in ln and "TIME_CEILING_HOURS" in ln
        for ln in result.stdout.splitlines()
    ), result.stdout


def test_slurm_duration_parser_covers_the_documented_formats():
    """Slurm accepts M, M:S, H:M:S, D-H, D-H:M and D-H:M:S.

    Extracted from the script rather than reimplemented, so this pins the real
    parser: a wrong reading here silently mis-reports the wall-clock margin.
    """
    src = open(SCRIPT, encoding="utf-8").read()
    body = src.split("def parse_slurm_duration(text):")[1].split("\nceiling_h = None")[0]
    namespace = {}
    exec("def parse_slurm_duration(text):" + body, namespace)  # noqa: S102
    parse = namespace["parse_slurm_duration"]

    assert parse("24:00:00") == 24.0
    assert parse("08:00:00") == 8.0
    assert parse("1-00:00:00") == 24.0
    assert parse("1-00") == 24.0
    assert parse("2-12") == 60.0
    assert parse("2-12:30") == 60.5
    assert parse("30") == 0.5  # bare minutes
    assert parse("90") == 1.5
    assert abs(parse("10:30") - 10.5 / 60) < 1e-9  # M:S
    # Unparseable input must be reported as unknown, never guessed at.
    assert parse("") is None
    assert parse("abc") is None
    assert parse("x:y") is None


# ---------------------------------------------------------------------------
# INIT_PARAM interactions
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "model.pt")),
    reason="no local checkpoint to point INIT_PARAM at",
)
def test_smoke_drops_init_param_but_keeps_model_conf_overrides(gpu_stub, tmp_path):
    """SMOKE proves the container, not a training lineage."""
    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "SMOKE": "1",
            "OUTPUT_DIR_SMOKE": str(tmp_path / "smoke"),
            "INIT_PARAM": os.path.join(MODEL_DIR, "model.pt"),
            "RICH_WEIGHT": "0.3",
            "EMO_MASK_TOKEN_ID": SER_TOKEN_ID,
        },
    )
    command = resolved_command(result.stdout)

    assert INIT_PARAM_FLAG not in command
    assert "ignoring INIT_PARAM" in result.stdout, "the drop must be announced"
    # The whole point of smoking this change:
    assert command_flag_values(command, RICH_FLAG) == ["0.3"]
    assert command_flag_values(command, EMO_FLAG) == [SER_TOKEN_ID]


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "model.pt")),
    reason="no local checkpoint to point INIT_PARAM at",
)
@pytest.mark.parametrize("checkpoint_present", [False, True])
def test_init_param_with_resume_warns_and_says_which_wins(
    gpu_stub, tmp_path, checkpoint_present
):
    """RESUME defaults to true, so this combination is normal, not an error.

    funasr applies init_param at model build and Trainer.resume_checkpoint
    overwrites it afterwards -- but only if OUTPUT_DIR/model.pt actually exists.
    The preflight must say which of the two supplies the weights, and must never
    fail on the combination (failing would make INIT_PARAM unusable without also
    passing RESUME=false).
    """
    out = tmp_path / "out"
    out.mkdir()
    if checkpoint_present:
        (out / "model.pt").write_bytes(b"not a real checkpoint")

    result = run_script(
        SCRIPT,
        gpu_stub,
        env={
            "INIT_PARAM": os.path.join(MODEL_DIR, "model.pt"),
            "RESUME": "true",
            "OUTPUT_DIR": str(out),
        },
    )

    # Never fatal.
    assert not any(
        "FAIL:" in ln and "INIT_PARAM" in ln for ln in result.stdout.splitlines()
    ), result.stdout

    notes = [ln for ln in result.stdout.splitlines() if "INIT_PARAM is set" in ln]
    assert len(notes) == 1, f"expected one note about the interaction, got {notes}"

    if checkpoint_present:
        assert "RESUMED checkpoint wins" in notes[0]
    else:
        assert "weights come from INIT_PARAM" in notes[0]

    # Either way the flag is still emitted; funasr decides at runtime.
    assert INIT_PARAM_FLAG in resolved_command(result.stdout)
