"""Offline tests for install.py logic and the flux-pack import resolution in
flux_fisheye_filling_nodes.py. Stubs pip/git so nothing is actually installed.

Run: python notebooks/test_install_logic.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = []


def ok(name):
    PASS.append(name)
    print(f"[ ok ] {name}")


def load_install_module():
    spec = importlib.util.spec_from_file_location(
        "camera_install", os.path.join(REPO, "install.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_commands(mod):
    """Replace subprocess-based helpers with recorders."""
    calls = []
    mod._run = lambda cmd, cwd=None: calls.append((tuple(cmd), cwd))
    mod._pip_install = lambda *args: calls.append((("pip",) + args, None))
    mod._git = lambda: "git"
    return calls


def test_sharp_early_return():
    mod = load_install_module()
    calls = stub_commands(mod)
    mod.ensure_sharp_checkout()  # real checkout has submodules/ml-sharpt/src/sharp
    assert calls == [], f"expected no commands, got {calls}"
    ok("sharp checkout present -> no git calls")


def test_sharp_clone_fallback():
    mod = load_install_module()
    calls = stub_commands(mod)
    with tempfile.TemporaryDirectory() as tmp:
        mod.NODE_DIR = os.path.join(tmp, "custom_nodes", "camera-comfyUI")
        os.makedirs(mod.NODE_DIR)
        # No .git -> should go straight to direct clone
        mod.ensure_sharp_checkout()
    assert len(calls) == 1 and "clone" in calls[0][0], calls
    assert mod.SHARP_GIT_URL in calls[0][0], calls
    ok("sharp missing + no .git -> direct clone")


def test_vggt_uses_https_git():
    mod = load_install_module()
    calls = stub_commands(mod)
    mod._importable = lambda name: False
    mod.ensure_vggt()
    assert calls == [(("pip", "vggt @ git+https://github.com/facebookresearch/vggt.git"), None)], calls
    ok("vggt -> pip install from git+https (no ssh)")


def test_flux_pack_skip_outside_comfyui():
    mod = load_install_module()
    calls = stub_commands(mod)
    with tempfile.TemporaryDirectory() as tmp:
        mod.NODE_DIR = os.path.join(tmp, "somewhere", "camera-comfyUI")
        os.makedirs(mod.NODE_DIR)
        mod.ensure_flux_inpainting_pack()
    assert calls == [], calls
    ok("flux pack: skipped when parent is not custom_nodes")


def test_flux_pack_detects_existing_and_clones_when_missing():
    mod = load_install_module()
    for existing in ("inpainting_flux", "ComfyUI-Flux-Inpainting", "ComfyUI-Flux-Inpainting-main"):
        calls = stub_commands(mod)
        with tempfile.TemporaryDirectory() as tmp:
            cn = os.path.join(tmp, "custom_nodes")
            mod.NODE_DIR = os.path.join(cn, "camera-comfyUI")
            os.makedirs(mod.NODE_DIR)
            os.makedirs(os.path.join(cn, existing))
            mod.ensure_flux_inpainting_pack()
        assert calls == [], f"{existing}: {calls}"
    ok("flux pack: all existing folder names detected, no clone")

    calls = stub_commands(mod)
    with tempfile.TemporaryDirectory() as tmp:
        cn = os.path.join(tmp, "custom_nodes")
        mod.NODE_DIR = os.path.join(cn, "camera-comfyUI")
        os.makedirs(mod.NODE_DIR)
        mod.ensure_flux_inpainting_pack()
        assert len(calls) == 1 and "clone" in calls[0][0], calls
        assert calls[0][0][-1] == os.path.join(cn, "inpainting_flux"), calls
    ok("flux pack: missing -> cloned as custom_nodes/inpainting_flux")


def test_main_never_fails():
    mod = load_install_module()

    def boom():
        raise RuntimeError("no network")

    mod.STEPS = (("step-a", boom), ("step-b", boom))
    assert mod.main() == 0
    ok("main() returns 0 even when every step fails")


def test_flux_import_resolution():
    """Build a fake ComfyUI tree and check _import_flux_inpainting finds the
    pack under a dashed folder name and honors its relative imports."""
    if "PIL" not in sys.modules:
        try:
            import PIL  # noqa: F401
        except ImportError:
            pil = types.ModuleType("PIL")
            pil.Image = types.SimpleNamespace()
            sys.modules["PIL"] = pil
            sys.modules["PIL.Image"] = types.ModuleType("PIL.Image")

    tmp = tempfile.mkdtemp()
    try:
        cn = os.path.join(tmp, "ComfyUI", "custom_nodes")
        pack = os.path.join(cn, "campack")
        os.makedirs(pack)
        for fname in ("reprojection_nodes.py", "flux_fisheye_filling_nodes.py"):
            shutil.copy(os.path.join(REPO, fname), pack)
        open(os.path.join(pack, "__init__.py"), "w").close()

        # Fake flux pack under a dashed (non-identifier) folder name with a
        # relative import, mirroring the real repo layout.
        flux = os.path.join(cn, "ComfyUI-Flux-Inpainting")
        os.makedirs(os.path.join(flux, "modules"))
        open(os.path.join(flux, "__init__.py"), "w").close()
        open(os.path.join(flux, "modules", "__init__.py"), "w").close()
        with open(os.path.join(flux, "modules", "load_util.py"), "w") as f:
            f.write("MARKER = 'loaded'\n")
        with open(os.path.join(flux, "nodes.py"), "w") as f:
            f.write(
                "from .modules.load_util import MARKER\n"
                "class FluxNF4Inpainting:\n"
                "    marker = MARKER\n"
            )

        sys.path.insert(0, os.path.dirname(pack))
        mod = importlib.import_module("campack.flux_fisheye_filling_nodes")
        assert mod._flux_import_error is None, mod._flux_import_error
        assert mod.FluxInpainting is not None
        assert mod.FluxInpainting.marker == "loaded"
        ok("flux import: dashed folder name resolved incl. relative imports")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_sharp_early_return()
    test_sharp_clone_fallback()
    test_vggt_uses_https_git()
    test_flux_pack_skip_outside_comfyui()
    test_flux_pack_detects_existing_and_clones_when_missing()
    test_main_never_fails()
    test_flux_import_resolution()
    print(f"\n{len(PASS)} install-logic checks passed")
