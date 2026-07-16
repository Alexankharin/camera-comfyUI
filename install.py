"""Post-install setup for camera-comfyUI.

ComfyUI-Manager runs this script automatically after installing the node pack
(both git and registry installs), right after `pip install -r requirements.txt`.
Manual users can run it themselves: `python install.py` from this directory.

It makes the optional heavy dependencies work out of the box:
  * vggt   — pip-installed from GitHub (not on PyPI); needed by VideoPoseEstimator.
  * SHARP  — the submodules/ml-sharpt checkout; needed by ImageToSplat & co.
  * gsplat — SHARP's rasterizer backend (JIT-compiles CUDA kernels on first use).
  * inpainting_flux — the ComfyUI-Flux-Inpainting node pack, cloned as a sibling
    custom node; needed by OutpaintAnyProjection / SplatTrajectoryEnricher.

Every step is idempotent and non-fatal: the node pack degrades gracefully at
runtime (see __init__.py), so a failed optional step only disables the nodes
that need it. The script therefore always exits 0 and reports what it skipped.
"""

import os
import shutil
import subprocess
import sys

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PREFIX = "[camera-comfyUI install]"

VGGT_GIT_URL = "https://github.com/facebookresearch/vggt.git"
SHARP_GIT_URL = "https://github.com/apple/ml-sharp"
FLUX_PACK_GIT_URL = "https://github.com/rubi-du/ComfyUI-Flux-Inpainting.git"
# Folder names under custom_nodes/ that count as "the flux inpainting pack is
# already installed" (must stay in sync with flux_fisheye_filling_nodes.py).
FLUX_PACK_CANDIDATES = (
    "inpainting_flux",
    "ComfyUI-Flux-Inpainting",
    "ComfyUI-Flux-Inpainting-main",
    "comfyui-flux-inpainting",
)


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def _run(cmd, cwd=None) -> None:
    _log("$ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def _pip_install(*args: str) -> None:
    _run([sys.executable, "-m", "pip", "install", *args])


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "git executable not found on PATH; cannot fetch GitHub dependencies"
        )
    return git


def _importable(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def ensure_base_requirements() -> None:
    """Install requirements.txt if it clearly has not been installed yet.

    ComfyUI-Manager installs it before running this script, so this only fires
    for manual `python install.py` users.
    """
    if _importable("diffusers") and _importable("transformers"):
        _log("base requirements already satisfied")
        return
    _pip_install("-r", os.path.join(NODE_DIR, "requirements.txt"))


def ensure_vggt() -> None:
    """VGGT (VideoPoseEstimator). Not on PyPI — install from GitHub over https."""
    if _importable("vggt"):
        _log("vggt already installed")
        return
    _pip_install(f"vggt @ git+{VGGT_GIT_URL}")
    _log("vggt installed from GitHub")


def ensure_sharp_checkout() -> None:
    """Materialize the SHARP submodule (ImageToSplat / VideoToFusedSplats).

    Registry archives already bundle it; git installs need `submodule update`
    (ComfyUI-Manager does not clone recursively). If this directory is not a
    git checkout at all, fall back to a direct clone.
    """
    sharp_dir = os.path.join(NODE_DIR, "submodules", "ml-sharpt")
    sharp_src = os.path.join(sharp_dir, "src", "sharp")
    if os.path.isdir(sharp_src):
        _log("SHARP checkout already present")
        return

    git = _git()
    if os.path.exists(os.path.join(NODE_DIR, ".git")):
        _run([git, "submodule", "update", "--init", "--recursive"], cwd=NODE_DIR)
        if os.path.isdir(sharp_src):
            _log("SHARP submodule initialized")
            return

    if os.path.isdir(sharp_dir) and os.listdir(sharp_dir):
        raise RuntimeError(
            f"{sharp_dir} exists but does not contain src/sharp; "
            "remove it and re-run install.py"
        )
    _run([git, "clone", "--depth", "1", SHARP_GIT_URL, sharp_dir])
    _log("SHARP cloned from GitHub")


def ensure_gsplat() -> None:
    """gsplat backs SHARP's import chain and the fast splat render path.

    pip install is lightweight (CUDA kernels JIT-compile on first use), but it
    can still fail on exotic setups — that only disables the SHARP/gsplat nodes.
    """
    if _importable("gsplat"):
        _log("gsplat already installed")
        return
    _pip_install("gsplat")
    _log("gsplat installed")


def ensure_flux_inpainting_pack() -> None:
    """Clone ComfyUI-Flux-Inpainting next to this pack if no copy exists yet."""
    custom_nodes_dir = os.path.dirname(NODE_DIR)
    if os.path.basename(custom_nodes_dir).lower() != "custom_nodes":
        _log(
            "not installed under a ComfyUI custom_nodes directory; "
            "skipping ComfyUI-Flux-Inpainting setup"
        )
        return

    for name in FLUX_PACK_CANDIDATES:
        if os.path.isdir(os.path.join(custom_nodes_dir, name)):
            _log(f"flux inpainting pack already present ({name})")
            return

    git = _git()
    target = os.path.join(custom_nodes_dir, "inpainting_flux")
    _run([git, "clone", "--depth", "1", FLUX_PACK_GIT_URL, target])
    pack_requirements = os.path.join(target, "requirements.txt")
    if os.path.isfile(pack_requirements):
        _pip_install("-r", pack_requirements)
    _log("ComfyUI-Flux-Inpainting installed as custom_nodes/inpainting_flux")


STEPS = (
    ("base requirements", ensure_base_requirements),
    ("vggt (VideoPoseEstimator)", ensure_vggt),
    ("SHARP checkout (ImageToSplat)", ensure_sharp_checkout),
    ("gsplat (splat rasterizer)", ensure_gsplat),
    ("ComfyUI-Flux-Inpainting (OutpaintAnyProjection)", ensure_flux_inpainting_pack),
)


def main() -> int:
    failures = []
    for title, step in STEPS:
        _log(f"--- {title} ---")
        try:
            step()
        except Exception as exc:  # keep going: each dependency is optional
            failures.append((title, exc))
            _log(f"WARNING: {title} failed: {exc}")

    if failures:
        _log("finished with warnings - the affected optional nodes stay disabled:")
        for title, exc in failures:
            _log(f"  * {title}: {exc}")
        _log("re-run `python install.py` after fixing the issue (network/git/pip)")
    else:
        _log("all optional dependencies are ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
