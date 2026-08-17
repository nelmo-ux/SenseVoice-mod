"""Why the text labeller has its own image, written down as assertions.

``docker/Dockerfile.textlabel`` exists because ``scripts/label_emotions_text.py``
needs vLLM, and vLLM's dependency set and ``requirements.txt``'s numpy ceiling
have no common solution.  That is easy to forget and expensive to rediscover: a
contributor who tries to fold the labeller into ``docker/Dockerfile.cluster``
finds out during a multi-minute build, from a resolver backtrack that names
neither numpy nor the reason for the cap.

So the conflict is asserted here instead, as arithmetic over the declared
version specifiers.  The numbers below were read from PyPI metadata and are
quoted in the Dockerfile's header for the same reason.

None of this needs vLLM, numpy 2.x or a GPU to run -- it compares specifiers,
never installs anything.
"""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile.textlabel"
REQUIREMENTS = ROOT / "requirements.txt"

#: Requirements vllm 0.27.1 declares that decide the split, read from its PyPI
#: metadata.  ``opencv-python-headless`` is the one that carries numpy: every
#: release from 4.13.0.90 up declares ``numpy>=2`` on Python >= 3.9, so vLLM
#: reaches numpy 2 transitively no matter which of those releases is chosen.
VLLM_REQUIRES = {
    "numba": "==0.65.0",
    "opencv-python-headless": ">=4.13.0",
}
OPENCV_REQUIRES_NUMPY = ">=2"

#: Real numpy releases spanning both sides of the ceiling.  A specifier
#: intersection can be expressed without being empty of *released* versions, so
#: the test below asks the question that matters -- is there a numpy anyone can
#: actually install that satisfies both -- rather than reasoning about the
#: ranges abstractly.
NUMPY_RELEASES = [
    "1.22.4",
    "1.24.4",
    "1.26.0",
    "1.26.4",
    "2.0.0",
    "2.1.3",
    "2.2.6",
    "2.3.4",
]

NUMBA_RELEASES = ["0.58.1", "0.59.1", "0.60.0", "0.61.0", "0.65.0"]


def _requirements():
    """Parse requirements.txt into ``{normalised name: Requirement}``.

    Commented-out optional extras are skipped: they are documentation of an
    opt-in install, not part of the default set whose numpy ceiling is at issue.
    """
    parsed = {}
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        requirement = Requirement(line)
        parsed[re.sub(r"[-_.]+", "-", requirement.name).lower()] = requirement
    return parsed


def _instructions():
    """Yield the Dockerfile's logical instructions, comments removed.

    Continuation lines are joined so that a ``pip install`` spread over several
    lines is one instruction, and full-line comments are dropped so that prose
    quoting a command is never mistaken for the command.
    """
    joined = []
    buffer = ""
    for raw in DOCKERFILE.read_text().splitlines():
        if raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


# --------------------------------------------------------------- the conflict


def test_no_released_numpy_satisfies_both_the_ceiling_and_vllm():
    """The split is arithmetic, not a preference.

    If this ever passes trivially because the ceiling moved, the images may be
    mergeable again -- but that is a decision to make deliberately, after
    rereading why the caps in requirements.txt exist (commit 319195d).
    """
    ceiling = _requirements()["numpy"].specifier
    both = [v for v in NUMPY_RELEASES if v in ceiling and v in SpecifierSet(OPENCV_REQUIRES_NUMPY)]
    assert not both, (
        f"numpy {both} appears to satisfy both requirements.txt ({ceiling}) and "
        f"vLLM's transitive numpy{OPENCV_REQUIRES_NUMPY}; if that is real, the "
        "reason docker/Dockerfile.textlabel is a separate image no longer holds "
        "and both the Dockerfile header and docs/chunk_training.md need updating."
    )


def test_no_released_numba_satisfies_both_the_cap_and_vllm():
    """The same conflict, one layer down and stated in vLLM's own metadata.

    numpy is reached through opencv; numba is a direct vllm requirement, so this
    one fails at resolution rather than at ABI time.
    """
    cap = _requirements()["numba"].specifier
    vllm_pin = SpecifierSet(VLLM_REQUIRES["numba"])
    both = [v for v in NUMBA_RELEASES if v in cap and v in vllm_pin]
    assert not both, (
        f"numba {both} satisfies both requirements.txt ({cap}) and vllm "
        f"({vllm_pin}); the numba cap is one of the pins holding the numpy "
        "ceiling, so check what moved before assuming the images can be merged."
    )


def test_requirements_still_holds_the_numpy_ceiling():
    """The two tests above are only meaningful while these caps are real."""
    requirements = _requirements()
    assert "2.0.0" not in requirements["numpy"].specifier, (
        "requirements.txt no longer excludes numpy 2.x; the caps on scipy, "
        "numba, llvmlite, scikit-learn and umap-learn exist only to hold that "
        "ceiling and should be revisited together with it."
    )
    for name in ["scipy", "numba", "llvmlite", "scikit-learn", "umap-learn"]:
        assert name in requirements, f"requirements.txt lost the {name} cap"


def test_training_requirements_do_not_pull_in_the_labeller_stack():
    """vLLM must not appear in the shared requirements file.

    Adding it there would put the two stacks in one install set, which is the
    exact thing the arithmetic above says cannot work -- and it would break the
    training image, which is the one with a schedule attached.
    """
    requirements = _requirements()
    for name in ["vllm", "opencv-python-headless"]:
        assert name not in requirements, (
            f"requirements.txt now requires {name}, which conflicts with its own "
            "numpy ceiling; the text labeller's dependencies belong in "
            "docker/Dockerfile.textlabel, not here."
        )


# ------------------------------------------------------------- the Dockerfile


def test_textlabel_image_does_not_install_the_training_requirements():
    """What is forbidden is *acting on* the file, not naming it.

    The header explains the conflict at length and one build-time assertion
    quotes the ceiling in its failure message; both are the opposite of
    installing it.  So this looks for the two instructions that would actually
    bring the training stack in.
    """
    for instruction in _instructions():
        assert "-r requirements.txt" not in instruction, (
            f"docker/Dockerfile.textlabel installs requirements.txt: {instruction!r}. "
            "That would drag the numpy<=1.26.4 ceiling in alongside vLLM."
        )
        assert not (
            instruction.split(None, 1)[0].upper() in {"COPY", "ADD"}
            and "requirements.txt" in instruction
        ), f"docker/Dockerfile.textlabel copies requirements.txt in: {instruction!r}"


def test_textlabel_image_pins_vllm_and_transformers_and_tags_the_vllm_version():
    """A tag that names a version the image does not contain is a trap.

    Dockerfile.cluster documents the same hazard for its own tag: a stale tag
    that still resolves fails quietly with the wrong contents rather than
    loudly at pull time.
    """
    text = DOCKERFILE.read_text()
    pinned = re.search(r"^\s*vllm==([0-9][^\s\\]*)", text, re.MULTILINE)
    assert pinned, "docker/Dockerfile.textlabel does not pin an exact vllm version"
    assert re.search(r"^\s*transformers==[0-9]", text, re.MULTILINE), (
        "docker/Dockerfile.textlabel does not pin an exact transformers version"
    )
    version = pinned.group(1)
    assert f"sensevoice-textlabel:vllm{version}-" in text, (
        f"the image installs vllm=={version} but no tag in the header names "
        f"vllm{version}; the tag and the pin have to move together."
    )


def test_textlabel_image_does_not_bake_the_qwen_weights():
    """~28 GB of weights in a layer, against a tight cluster quota.

    The weights are staged on the shared filesystem and passed with --model, so
    nothing in this image should be fetching or copying them.
    """
    downloaders = [
        "huggingface-cli download",
        "hf download",
        "snapshot_download",
        "modelscope download",
        "git lfs",
    ]
    for instruction in _instructions():
        for downloader in downloaders:
            assert downloader not in instruction, (
                f"docker/Dockerfile.textlabel runs {downloader!r}; the Qwen2.5 "
                "weights are staged on the cluster and passed via --model, not "
                "baked into the image."
            )
        assert not re.search(r"\.safetensors|\.bin\b", instruction), (
            f"docker/Dockerfile.textlabel appears to copy model weights: {instruction!r}"
        )


def test_pip_install_happens_only_in_build_layers():
    """Never at runtime, never on a login node.

    A pip install reached from an ENTRYPOINT or CMD would run once per job, on
    whatever node the scheduler picked, against whatever egress it has.
    """
    for instruction in _instructions():
        if "pip install" not in instruction:
            continue
        assert instruction.split(None, 1)[0].upper() == "RUN", (
            f"pip install outside a build layer: {instruction!r}"
        )


def test_textlabel_image_verifies_itself_before_the_scheduler_does():
    """The build-layer checks are the point, not decoration.

    Dockerfile.cluster's equivalent is what caught its missing torchaudio before
    any job was queued.  This image's version proves the vLLM import, the
    Qwen2.5 chat-template stack and the labeller module, which are the ways a
    passing build can still fail hours later on the node -- and the scheduler
    discards the batch script's exit code, so that failure is reported as a
    successful job.  These layers are the last place it can be seen.
    """
    text = DOCKERFILE.read_text()
    for marker in [
        "from vllm import LLM, SamplingParams",
        "import label_emotions_text",
        "inspect.signature(LLM.chat)",
        'AutoConfig.for_model("qwen2")',
    ]:
        assert marker in text, (
            f"docker/Dockerfile.textlabel no longer verifies {marker!r} at build "
            "time; without it an ImportError surfaces only after the scheduler "
            "has started the job."
        )
