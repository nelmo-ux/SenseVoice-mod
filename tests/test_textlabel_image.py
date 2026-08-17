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

The second half of this file is about the toolchain, and it is here because of
what the first nine tests did *not* pin.  The v1 image passed every one of them,
built clean, pushed clean, and then died on the node: vLLM's Triton JIT compiles
CUDA utilities at first engine start by shelling out to a C compiler, and a CUDA
*runtime* base image ships none.

Worth being blunt about it, because the lesson is in the selection and not in
any individual test: not one of the nine would have caught it, and none of them
was wrong.  Four assert that the image installs the right versions, two that it
does not install the wrong ones, one that pip stays in build layers, one that
the weights are not baked in, and one that four verification layers are still
present -- and that last one checks their *markers*, which were all imports.
Nine properties, all of them about what the image contains.  The property that
failed was about what it can do.

The pattern is worth naming, because it is not specific to compilers: a test
written from an assumption tends to check the assumption rather than the
artefact.  "vLLM is installed" was the assumption; "vLLM can start an engine"
was the artefact, and nothing here or in the build touched it.  The cheap
correction, applied below, is to make at least one check *use* the thing instead
of describing it -- so the tests that follow ask that a compiler is installed,
and that the Dockerfile proves it by compiling, linking and running something
rather than by finding a binary on PATH.

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
    """~28 GB of weights re-pulled by every node that runs the image.

    The quota has room for them; what it does not have is a reason to spend that
    room.  The weights are staged on the shared filesystem and passed with
    --model, so nothing in this image should be fetching or copying them.
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


# ----------------------------------------------------------------- the toolchain


def test_textlabel_image_installs_a_c_compiler():
    """The v1 failure, asserted so it cannot come back quietly.

    vLLM and Triton compile CUDA utilities with a runtime JIT that shells out to
    a C compiler, and torch.compile's inductor backend shells out to a C++ one.
    The base is nvidia/cuda:*-runtime and has neither, so this has to be added
    explicitly -- and it looks like dead weight to anyone reading the apt line
    without the header, which is exactly why it is pinned here.
    """
    apt = " ".join(i for i in _instructions() if "apt-get install" in i)
    assert apt, "docker/Dockerfile.textlabel no longer installs any apt packages"
    assert re.search(r"\b(build-essential|gcc|clang)\b", apt), (
        "docker/Dockerfile.textlabel installs no C compiler. vLLM imports fine "
        "without one -- it needs it at first engine start, on the node, hours "
        "into the queue. This is the v1 failure; see the Dockerfile header."
    )
    assert re.search(r"\bpython3-dev\b", apt), (
        "docker/Dockerfile.textlabel no longer installs python3-dev. Triton's "
        "JIT-compiled module opens with #include <Python.h>, and Ubuntu ships "
        "that header in python3-dev rather than python3, so a compiler on its "
        "own is not enough."
    )


def test_the_toolchain_check_compiles_something_rather_than_locating_a_binary():
    """`which cc` proves a file is on PATH, which is not the failing property.

    A compiler that is present but cannot produce a runnable binary -- missing
    libc headers, no linker, a broken install -- passes an existence check and
    fails the job.  So the probe has to compile, link and then *run* its output,
    and this test pins all three parts rather than the string ``cc``.
    """
    probes = [i for i in _instructions() if "ccprobe" in i]
    assert probes, (
        "docker/Dockerfile.textlabel has no compile probe. Without it a missing "
        "or broken toolchain is discovered by the scheduler, and the scheduler "
        "discards the batch script's exit code."
    )
    compiled = [i for i in probes if "-x c" in i]
    assert compiled, (
        "the compile probe in docker/Dockerfile.textlabel no longer compiles a "
        f"translation unit: {probes!r}"
    )
    for instruction in compiled:
        assert re.search(r"&&\s*/tmp/ccprobe\b", instruction), (
            "the compile probe builds /tmp/ccprobe but never runs it; a binary "
            "that will not execute still counts as a successful compile."
        )
    for instruction in _instructions():
        if re.search(r"\b(which|command\s+-v)\s+(cc|gcc|c\+\+|g\+\+)\b", instruction):
            assert "-x c" in instruction or "ccprobe" in instruction, (
                f"the toolchain is verified by an existence check: {instruction!r}. "
                "That is the check that would have passed on v1 too."
            )


def test_the_toolchain_check_follows_the_path_triton_takes():
    """The plain compile is the floor; this is the part that matches reality.

    Triton does not run ``cc hello.c``.  It resolves a compiler in its own order,
    finds Python.h through sysconfig (which inside this venv resolves to the
    system interpreter's headers, so it depends on python3-dev), and builds a
    CPython extension.  Each of those is a separate way to be broken while the
    plain probe still passes, so the second layer walks the real path and this
    test pins the pieces of it that can run without a GPU.
    """
    text = DOCKERFILE.read_text()
    for marker, why in [
        ("sysconfig.get_paths", "Triton locates Python.h through sysconfig"),
        ("posix_local", "Debian's sysconfig scheme has to be remapped as Triton remaps it"),
        ("Python.h", "the JIT-compiled module includes it and python3-dev supplies it"),
        ("triton.runtime.build", "the real build entry point is preferred over a reimplementation"),
        ("PyInit_ccprobe", "the probe builds an actual CPython extension"),
        ("spec_from_file_location", "and loads it, which is what proves the ABI matches"),
    ]:
        assert marker in text, (
            f"docker/Dockerfile.textlabel no longer checks {marker!r} at build "
            f"time ({why}); that part of the JIT path goes back to being first "
            "exercised on the node."
        )


def test_the_triton_probe_checks_the_call_it_is_about_to_make():
    """The probe's own version of the mistake it exists to catch.

    Its first version asserted that ``_build``'s parameter names were a superset
    of the six it knew about, then called it without ``ccflags``, which the
    installed Triton requires: the guard passed, the call raised TypeError, and
    the fallback was never reached.  A superset test answers "does this signature
    look familiar"; what has to hold is "the call I am about to make binds".

    So the invariant pinned here is that the keyword mapping which is *checked*
    is the mapping which is *passed* -- one dict, bound and then unpacked.  An
    inline keyword list at the call site is what allowed the two to drift apart.

    Worth noting how that failure went, because it is the property to keep: the
    probe crashed loudly during a build instead of quietly taking the fallback
    and reporting success.  It cost one cheap build rather than a GPU job, and a
    version that had silently degraded would have reported a pass while testing
    nothing.  The assertions below therefore also require the fallback to
    announce itself.
    """
    text = DOCKERFILE.read_text()
    assert "s.bind(**kw)" in text, (
        "the probe no longer validates the _build call with Signature.bind; "
        "whatever replaced it is describing the signature rather than trying "
        "the call, which is exactly how the ccflags TypeError got shipped."
    )
    assert "build(**kw)" in text, (
        "the probe binds one keyword mapping and calls _build with something "
        "else; the check and the call have to be the same dict or they drift."
    )
    assert not re.search(r"\bbuild\(name=", text), (
        "the _build call passes an inline keyword list again. That is the form "
        "that allowed the guard and the call to disagree -- assemble the kwargs "
        "once, bind them, then unpack them."
    )
    assert "ccflags" in text, (
        "the probe no longer passes ccflags. The Triton in this image requires "
        "it, and omitting it is what failed the v2 build."
    )
    assert not re.search(r"<=\s*set\(inspect\.signature", text), (
        "the superset guard is back; it answers a different question from the "
        "one that matters (see this test's docstring)."
    )
    fallback = [i for i in _instructions() if "subprocess.check_call" in i]
    assert fallback, (
        "the direct-invocation fallback is gone, so a Triton that renames or "
        "drops _build would fail the build rather than compile a weaker probe."
    )
    assert "NOT exercised" in text and "WARNING" in text, (
        "the fallback no longer announces itself. A probe that quietly settles "
        "for the weaker check reports a pass while testing less than it claims."
    )


def test_the_image_tag_names_the_rebuilt_image():
    """v1 is the image that fails on the node, so nothing may still request it.

    Only tags are checked, not the string ``-v1``: the header discusses the v1
    incident in prose deliberately, and that prose is the reason the toolchain
    below it is not mistaken for bloat.
    """
    tags = re.findall(r"sensevoice-textlabel:vllm[0-9][^\s\\]*", DOCKERFILE.read_text())
    assert tags, "docker/Dockerfile.textlabel names no image tag at all"
    stale = [t for t in tags if not t.endswith("-v2")]
    assert not stale, (
        f"docker/Dockerfile.textlabel still names {stale}; v1 is the build with "
        "no C compiler in it, and a tag that still resolves in the registry "
        "fails quietly with those contents rather than loudly at pull time."
    )
