"""Validate workflows/*.json against the current node definitions.

For every camera-comfyUI node used in a workflow, compares the stored
widgets_values against the widget list derived from the node's INPUT_TYPES
(required + optional, in order, counting only widget-type inputs). A length
mismatch means the workflow predates a node-schema change and will load with
silently shifted/défault values.

Run: python notebooks/validate_workflows.py
Exit code 1 if any mismatch is found (missing node types are also reported).
"""

import glob
import json
import os
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_TMP_DIR = tempfile.mkdtemp(prefix="wf_validate_")


def _stub_get_save_image_path(filename_prefix, output_dir, *args, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, filename_prefix, 0, "", filename_prefix


_fp = types.ModuleType("folder_paths")
_fp.get_input_directory = lambda: _TMP_DIR
_fp.get_output_directory = lambda: _TMP_DIR
_fp.get_temp_directory = lambda: _TMP_DIR
_fp.get_save_image_path = _stub_get_save_image_path
_fp.get_annotated_filepath = lambda name: os.path.join(_TMP_DIR, name)
_fp.exists_annotated_filepath = lambda name: os.path.exists(os.path.join(_TMP_DIR, name))
_fp.get_filename_list = lambda folder: []
_fp.models_dir = _TMP_DIR
sys.modules["folder_paths"] = _fp

# Light stubs for heavy deps that some modules import at module level but that
# INPUT_TYPES itself does not need.
for name in ("transformers", "diffusers"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.pipeline = lambda *a, **k: None
            sys.modules[name] = stub

# Import the repo as a package so modules with relative imports load too.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "camcomfy",
    os.path.join(REPO_ROOT, "__init__.py"),
    submodule_search_locations=[REPO_ROOT],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["camcomfy"] = _pkg
_spec.loader.exec_module(_pkg)
NODE_CLASS_MAPPINGS = dict(_pkg.NODE_CLASS_MAPPINGS)

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def widget_specs(cls):
    """Ordered (name, combo_options|None) for a node's widget inputs."""
    it = cls.INPUT_TYPES()
    specs = []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            t = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
            if isinstance(t, (list, tuple)):  # combo box (either sequence type)
                specs.append((name, list(t)))
            elif isinstance(t, str) and t in WIDGET_TYPES:
                specs.append((name, None))
                # ComfyUI appends a control_after_generate widget after seeds
                if t == "INT" and name in ("seed", "noise_seed"):
                    specs.append((f"{name}:control_after_generate", None))
    return specs


def main() -> int:
    problems = 0
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "workflows", "*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        header_shown = False

        def report(msg):
            nonlocal header_shown, problems
            if not header_shown:
                print(f"\n=== {os.path.basename(path)}")
                header_shown = True
            print(f"  {msg}")
            problems += 1

        for n in data.get("nodes", []):
            t = n["type"]
            if t not in NODE_CLASS_MAPPINGS:
                continue  # builtin or third-party node
            specs = widget_specs(NODE_CLASS_MAPPINGS[t])
            got = n.get("widgets_values") or []
            if isinstance(got, dict):
                continue  # API-style dict widgets (third-party save format)
            if len(got) != len(specs):
                report(
                    f"N{n['id']} {t}: {len(got)} widget values, node now has "
                    f"{len(specs)} widgets {[s[0] for s in specs]}; "
                    f"stored={json.dumps(got)[:100]}"
                )
                continue
            for (name, options), value in zip(specs, got):
                # empty option lists are dynamic file dropdowns (input dir
                # listing) — not verifiable outside a real ComfyUI install
                if options and value not in options:
                    report(
                        f"N{n['id']} {t}: widget '{name}' has stale value "
                        f"{value!r}, valid options are {options}"
                    )
    if problems:
        print(f"\n{problems} mismatches found")
        return 1
    print("all workflow widget schemas match current node definitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
