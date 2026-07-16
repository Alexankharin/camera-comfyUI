"""One-shot July 2026 workflow rework: schema repairs + documentation.

For every workflows/*.json this script:
  1. migrates widgets_values stored with pre-2026 node schemas to the current
     widget lists (see notebooks/validate_workflows.py for the detector);
  2. fixes broken references (wan UNET filename typo, ReprojectImage rotation
     widgets that no longer exist -> explicit TransformToMatrix nodes);
  3. adds an embedded MarkdownNote explaining purpose/stages/requirements,
     meaningful group boxes and node titles;
  4. rewrites the JSON pretty-printed (indent 2).

Idempotent-ish: repairs are guarded by length/value checks, notes are only
added if no MarkdownNote titled 'About this workflow' exists.

Run: python notebooks/rework_workflows_2026_07.py
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(REPO, "workflows")

DA_MODEL = "Depth-Anything-V2-Metric-Indoor-Base-hf"

NOTE_TITLE = "About this workflow"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def load(name):
    with open(os.path.join(WF, name), encoding="utf-8") as fh:
        return json.load(fh)


def save(name, data):
    # normalize pos/size dict-form (pre-2024 serialization) to arrays
    for n in data.get("nodes", []):
        for key in ("pos", "size"):
            v = n.get(key)
            if isinstance(v, dict):
                n[key] = [v[k] for k in sorted(v, key=lambda s: int(s))]
    with open(os.path.join(WF, name), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def node(data, nid):
    for n in data["nodes"]:
        if n["id"] == nid:
            return n
    raise KeyError(f"node {nid} not found")


def xy(v):
    if isinstance(v, dict):
        return [float(v["0"]), float(v["1"])]
    return [float(v[0]), float(v[1])]


def bbox(data, ids, pad_top=60, pad=20):
    xs, ys, xe, ye = [], [], [], []
    for nid in ids:
        n = node(data, nid)
        p, s = xy(n["pos"]), xy(n["size"])
        xs.append(p[0]); ys.append(p[1])
        xe.append(p[0] + s[0]); ye.append(p[1] + s[1])
    x, y = min(xs) - pad, min(ys) - pad_top
    return [x, y, max(xe) + pad - x, max(ye) + pad - y]


GROUP_COLORS = ["#3f789e", "#a1309b", "#8A8", "#b58b2a", "#88A", "#b06634", "#535"]


def set_groups(data, groups):
    """groups: list of (title, node_ids). Replaces the groups list."""
    out = []
    for i, (title, ids) in enumerate(groups):
        out.append({
            "id": i + 1,
            "title": title,
            "bounding": bbox(data, ids),
            "color": GROUP_COLORS[i % len(GROUP_COLORS)],
            "font_size": 24,
            "flags": {},
        })
    data["groups"] = out


def retitle_groups(data, titles):
    """titles: list matching data['groups'] order."""
    assert len(titles) == len(data.get("groups", [])), "group count mismatch"
    for g, t in zip(data["groups"], titles):
        g["title"] = t


def set_title(data, nid, title):
    node(data, nid)["title"] = title


def next_node_id(data):
    data["last_node_id"] = int(data.get("last_node_id", 0)) + 1
    return data["last_node_id"]


def next_link_id(data):
    data["last_link_id"] = int(data.get("last_link_id", 0)) + 1
    return data["last_link_id"]


def add_note(data, markdown, pos=None, size=None):
    for n in data["nodes"]:
        if n["type"] in ("MarkdownNote", "Note") and n.get("title") == NOTE_TITLE:
            n["widgets_values"] = [markdown]
            return n["id"]
    if pos is None:
        xs = [xy(n["pos"])[0] for n in data["nodes"]]
        ys = [xy(n["pos"])[1] for n in data["nodes"]]
        pos = [min(xs) - 560, min(ys)]
    if size is None:
        lines = markdown.count("\n") + 1
        size = [520, max(220, min(700, 26 * lines + 80))]
    nid = next_node_id(data)
    data["nodes"].append({
        "id": nid,
        "type": "MarkdownNote",
        "title": NOTE_TITLE,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": [markdown],
        "color": "#432",
        "bgcolor": "#653",
    })
    return nid


def add_transform_node(data, pos, widgets):
    nid = next_node_id(data)
    data["nodes"].append({
        "id": nid,
        "type": "TransformToMatrix",
        "pos": pos,
        "size": [315, 154],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": "transformation matrix", "type": "MAT_4X4",
                     "links": [], "slot_index": 0}],
        "properties": {"Node name for S&R": "TransformToMatrix"},
        "widgets_values": widgets,
    })
    return nid


def link(data, src, dst, dst_input, ltype="MAT_4X4", src_slot=0):
    """Connect src node output slot to dst node's named input (added if absent)."""
    dnode = node(data, dst)
    inputs = dnode.setdefault("inputs", [])
    slot = None
    for i, inp in enumerate(inputs):
        if inp["name"] == dst_input:
            slot = i
            break
    if slot is None:
        inputs.append({"name": dst_input, "type": ltype, "link": None})
        slot = len(inputs) - 1
    lid = next_link_id(data)
    inputs[slot]["link"] = lid
    snode = node(data, src)
    snode["outputs"][src_slot].setdefault("links", None)
    if snode["outputs"][src_slot]["links"] is None:
        snode["outputs"][src_slot]["links"] = []
    snode["outputs"][src_slot]["links"].append(lid)
    data["links"].append([lid, src, src_slot, dst, slot, ltype])
    return lid


# --------------------------------------------------------------------------- #
# Global widget-schema repairs (2025 -> 2026 node schemas)
# --------------------------------------------------------------------------- #

def repair_widgets(data):
    changed = []
    for n in data["nodes"]:
        t, w = n["type"], n.get("widgets_values")
        if not isinstance(w, list):
            continue
        if t == "DepthEstimatorNode" and len(w) == 2:
            n["widgets_values"] = w + [1]  # median_blur_kernel
        elif t == "FisheyeDepthEstimator" and len(w) == 6:
            # old: [model, scale, pfov, pres, fres, softmerge_radius]
            n["widgets_values"] = w[:5] + ["SOFTMERGE", w[5], 1]
        elif t == "PointCloudCleaner" and len(w) == 2:
            # old (voxel_size, min_points) were world-unit; semantics changed
            n["widgets_values"] = [1024, 1024, 1.0, 3]
        elif t == "CameraMotionNode" and len(w) == 6:
            n["widgets_values"] = w + [0, False, False]
        elif t == "CameraInterpolationNode" and len(w) == 0:
            n["widgets_values"] = [2]  # keyframes; renderers interpolate frames
        elif t == "PointcloudTrajectoryEnricher" and len(w) == 20:
            # old tail: reproject-back proj/fov/w/h + legacy backend/voxel params
            n["widgets_values"] = w[:13] + [0.07, 3, DA_MODEL]
        else:
            continue
        changed.append(f"{t}#{n['id']}")
    return changed


# --------------------------------------------------------------------------- #
# Per-file specs
# --------------------------------------------------------------------------- #

def do_demo_camera(name="demo_camera_workflow.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 1, "Rotate camera (pitch 60°)")
    set_title(d, 2, "Pinhole 90° → Equirect 180°")
    set_title(d, 7, "Reprojected image")
    set_title(d, 4, "Coverage mask (white = hole)")
    add_note(d, (
        "# Camera reprojection demo\n\n"
        "Minimal example of the two core nodes: **TransformToMatrix** rotates the "
        "virtual camera (theta = 60° pitch) and **ReprojectImage** converts a 90° "
        "pinhole image into a 180° equirectangular view.\n\n"
        "Previews show the reprojected image and the coverage mask "
        "(white = pixels the source image cannot see).\n\n"
        "- **Set:** the image in LoadImage.\n"
        "- **Requires:** nothing beyond this pack (no models).\n"
        "- **Try:** switch `output_projection` to FISHEYE, or raise `feathering` "
        "to soften the mask edge."
    ))
    save(name, d)


def do_outpaint_node_test(name="Outpaint_node_test.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 11, "Outpaint one patch (yaw +45°)")
    set_title(d, 3, "Result")
    set_title(d, 5, "Holes still to fill")
    add_note(d, (
        "# OutpaintAnyProjection smoke test\n\n"
        "Single-node sanity check: a 90° pinhole image is placed on a 180° fisheye "
        "canvas and one 90° pinhole patch at yaw +45° is Flux-inpainted "
        "(10 steps for speed).\n\n"
        "Outputs: the partially outpainted canvas and the *remaining holes* mask — "
        "chain more OutpaintAnyProjection nodes at other angles to fill it "
        "(see `Outpaint_fisheye180.json`).\n\n"
        "- **Set:** input image; prompt inside the node (empty = unconditional).\n"
        "- **Requires:** `custom_nodes/inpainting_flux` "
        "(installed automatically by this pack's install.py) — downloads "
        "FLUX.1-Fill NF4 weights on first run."
    ))
    save(name, d)


def do_fisheye_to_pointcloud(name="fisheye_to_pointcloud.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 155, "Metric depth (fisheye-aware)")
    set_title(d, 160, "Depth → point cloud")
    set_title(d, 161, "Save .ply / .npy")
    set_groups(d, [
        ("1. Fisheye metric depth", [154, 155, 156, 157, 158, 159]),
        ("2. Unproject & save", [160, 161]),
    ])
    add_note(d, (
        "# Fisheye 180° → point cloud\n\n"
        "Estimates metric depth directly on a 180° fisheye image — "
        "**FisheyeDepthEstimator** internally splits it into pinhole views, runs "
        "Depth-Anything V2 on each and merges the depths back (SOFTMERGE) — then "
        "unprojects image + depth to a 3D point cloud and saves it.\n\n"
        "Previews: colorized depth and the validity mask.\n\n"
        "- **Set:** fisheye input image (e.g. produced by "
        "`Outpaint_fisheye180.json`); filename in SavePointCloud.\n"
        "- **Requires:** Depth-Anything V2 (auto-downloads from HuggingFace).\n"
        "- **Next:** view the cloud with `test_pointcloud_loading.json` or "
        "synthesize a second eye with `sbs180_workflow.json`."
    ))
    save(name, d)


def do_pointcloud(name="PointCloud.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 6, "Move camera (dolly −0.1, yaw 20°)")
    set_title(d, 31, "Render novel view")
    set_title(d, 47, "Drop stretched/occluded points")
    set_title(d, 43, "Novel-view depth → image")
    set_groups(d, [
        ("1. Image → metric ray depth", [1, 38, 45, 42, 40]),
        ("2. Lift to 3D & move camera", [18, 6, 9, 47]),
        ("3. Novel view & previews", [31, 13, 12, 43, 44, 25, 26]),
    ])
    add_note(d, (
        "# Single image → point cloud → novel view\n\n"
        "Lifts one pinhole image to a 3D point cloud via monocular metric depth, "
        "moves the camera, cleans occlusion artifacts and re-renders from the new "
        "viewpoint.\n\n"
        "Stages: DepthEstimator → ZDepthToRayDepth → DepthToPointCloud → "
        "TransformPointCloud (dolly −0.1, yaw 20°) → PointCloudCleaner → "
        "ProjectPointCloud. MedianFilterImage smooths the re-render "
        "(from ComfyUI-Image-Filters, optional).\n\n"
        "- **Set:** input image; the camera move in TransformToMatrix.\n"
        "- **Requires:** Depth-Anything V2 (auto-download); ComfyUI-Image-Filters "
        "only for the median-filter preview.\n"
        "- **Note:** PointCloudCleaner params were reset to defaults during the "
        "2026-07 schema migration — retune voxel_size / min_points_per_voxel if "
        "the render looks too sparse."
    ))
    save(name, d)


def do_pointcloud_walker(name="Pointcloud walker.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 5, "Camera move (edit me)")
    set_title(d, 17, "Build trajectory")
    set_title(d, 14, "Render fly-through")
    set_groups(d, [
        ("1. Image → point cloud", [1, 3, 20, 2]),
        ("2. Trajectory", [5, 4, 17]),
        ("3. Render & save", [14, 10]),
    ])
    add_note(d, (
        "# Point cloud walker (fly-through video)\n\n"
        "Single image → metric depth → point cloud, then **CameraTrajectoryNode** "
        "derives a camera path and **CameraMotionNode** renders a fly-through "
        "saved as WEBM.\n\n"
        "- **Set:** input image; the move in TransformToMatrix "
        "(shift XYZ + theta/phi); frames-per-segment (`n_points`) in "
        "CameraMotionNode.\n"
        "- **Requires:** Depth-Anything V2 (auto-download).\n"
        "- **Note:** the pinhole FOV here is 60° — keep DepthToPointCloud and "
        "CameraMotionNode FOVs consistent with your input."
    ))
    save(name, d)


def do_test_pointcloud_loading(name="Test pointcloud_loading.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 6, "Load saved cloud (.ply/.npy)")
    set_title(d, 1, "Start pose (identity)")
    set_title(d, 5, "End pose (dolly +0.1)")
    set_title(d, 7, "Render motion")
    add_note(d, (
        "# Load a saved point cloud & orbit\n\n"
        "Reloads a point cloud saved by SavePointCloud and renders a short camera "
        "move between two poses (identity → 0.1 forward) as a WEBM.\n\n"
        "- **Set:** the file in LoadPointCloud (dropdown lists the ComfyUI input "
        "dir — run `PointCloud.json` or `fisheye_to_pointcloud.json` first); the "
        "two TransformToMatrix poses.\n"
        "- **Requires:** nothing beyond this pack."
    ))
    save(name, d)


def do_pc_enricher(name="PC_enricher.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 3, "Enrich cloud along trajectory (Flux outpaint)")
    set_title(d, 11, "Orbit preview of enriched cloud")
    set_groups(d, [
        ("1. Fisheye → point cloud", [13, 14, 17, 16, 15]),
        ("2. Enrich along trajectory", [18, 3]),
        ("3. Save & preview", [4, 11, 12]),
    ])
    add_note(d, (
        "# Point cloud enricher (outpaint along a trajectory)\n\n"
        "Fisheye image → metric depth → point cloud, then "
        "**PointcloudTrajectoryEnricher** walks the loaded camera trajectory: at "
        "each pose it renders the cloud, Flux-inpaints the disocclusion holes, "
        "re-estimates depth, aligns it and merges the new points into the cloud. "
        "The enriched cloud is saved and previewed as an orbit video.\n\n"
        "This is the one-node version of `pointcloud_inpaint.json`.\n\n"
        "- **Set:** input image; trajectory .npy (record one with "
        "SaveTrajectory); prompt inside the enricher.\n"
        "- **Requires:** inpainting_flux (auto-installed), Depth-Anything V2, "
        "FLUX.1-Fill NF4 weights.\n"
        "- **Note:** 2026-07 schema migration — the enricher's render/reproject "
        "options are now internal; voxel merge params were reset to defaults."
    ))
    save(name, d)


def do_fisheye_depth(name="Fisheye_depth_workflow.json"):
    d = load(name)
    repair_widgets(d)
    retitle_groups(d, [
        "View 1: center pinhole 90°",
        "View 2: yaw +45°",
        "View 3: yaw −45°",
        "View 4: pitch +45°",
        "View 5: pitch −45°",
        "View 6: full-fisheye fallback",
    ])
    # fusion chain + export were never grouped
    d["groups"].append({
        "id": 7,
        "title": "Fuse views & export point cloud",
        "bounding": bbox(d, [95, 146, 147, 148, 149, 105, 151, 153, 154]),
        "color": "#b58b2a",
        "font_size": 24,
        "flags": {},
    })
    set_title(d, 149, "Final fuse (SRC keeps fused views)")
    set_title(d, 151, "Fused depth → point cloud")
    add_note(d, (
        "# Multi-view fisheye depth — manual reference\n\n"
        "Estimates consistent metric depth over a 180° fisheye image by "
        "reprojecting it into five pinhole views (center, yaw ±45°, pitch ±45°), "
        "running Depth-Anything V2 on each, reprojecting the depths back to the "
        "fisheye and progressively fusing them (CombineDepthsNode + "
        "DepthRenormalizer to align scales), plus a full-fisheye pass for the "
        "rim. The fused depth is unprojected and saved as a point cloud.\n\n"
        "⚠️ **This whole graph is now one node** — `FisheyeDepthEstimator` does "
        "the same split-and-merge internally (see `fisheye_to_pointcloud.json`). "
        "Kept as a transparent, tweakable reference implementation.\n\n"
        "- **Set:** fisheye input image; SavePointCloud filename.\n"
        "- **Requires:** Depth-Anything V2 (auto-download)."
    ))
    save(name, d)


def do_outpaint_fisheye180(name="Outpaint_fisheye180.json"):
    d = load(name)
    repair_widgets(d)
    retitle_groups(d, [
        "2. Chained patch outpaints",
        "1. Pinhole → fisheye canvas",
    ])
    set_title(d, 2, "Outpaint yaw +45°")
    set_title(d, 3, "Outpaint yaw −45°")
    set_title(d, 4, "Outpaint pitch +45°")
    set_title(d, 5, "Outpaint pitch −45°")
    set_title(d, 9, "Full-frame pass (low-res rim fill)")
    set_title(d, 15, "Upscale rim fill to 4096")
    set_title(d, 12, "Composite sharp patches over rim")
    add_note(d, (
        "# Outpaint to a full 180° fisheye\n\n"
        "Places a 90° pinhole image onto a 180° fisheye canvas (ReprojectImage), "
        "then chains **five OutpaintAnyProjection passes** — yaw +45°, yaw −45°, "
        "pitch +45°, pitch −45°, and a low-res full-frame pass for the rim. Each "
        "pass consumes the previous pass's *remaining holes* mask. A PorterDuff "
        "composite keeps the sharp high-res patches on top of the upscaled rim "
        "fill.\n\n"
        "- **Set:** input image; prompts inside each outpaint node (optional).\n"
        "- **Requires:** inpainting_flux (auto-installed by install.py); "
        "FLUX.1-Fill NF4 weights download on first run (~12 GB VRAM).\n"
        "- **Output:** `Saved_fisheye` PNG — used as the input of "
        "`fisheye_to_pointcloud.json`, `sbs180_workflow.json` and "
        "`PC_enricher.json`."
    ))
    save(name, d)


def do_outpainting_fisheye_sd(name="outpainting_fisheye.json"):
    d = load(name)
    repair_widgets(d)
    # --- structural repair: pre-2025 ReprojectImage stored patch rotations as
    # widgets; the current node takes a MAT_4X4 transform_matrix input and an
    # inverse flag instead (mirrors outpainting_fisheye_flux.json).
    fixes = {
        42: ([90, 180, "PINHOLE", "FISHEYE", 4096, 4096, False, 0], None),
        40: ([180, 90, "FISHEYE", "PINHOLE", 1024, 1024, False, 0], "+42"),
        51: ([90, 180, "PINHOLE", "FISHEYE", 4096, 4096, True, 0], "+42"),
        59: ([180, 90, "FISHEYE", "PINHOLE", 1024, 1024, False, 0], "-42"),
        63: ([90, 180, "PINHOLE", "FISHEYE", 4096, 4096, True, 0], "-42"),
        67: ([180, 180, "FISHEYE", "FISHEYE", 1024, 1024, False, 0], None),
        72: ([180, 180, "FISHEYE", "FISHEYE", 4096, 4096, False, 0], None),
    }
    needs_repair = any(len(node(d, nid).get("widgets_values", [])) == 9
                       for nid in fixes)
    if needs_repair:
        m_pos = add_transform_node(d, [20, 480], [0, 0, 0, 0, 42])
        m_neg = add_transform_node(d, [20, 1180], [0, 0, 0, 0, -42])
        set_title(d, m_pos, "Patch rotation +42°")
        set_title(d, m_neg, "Patch rotation −42°")
        for nid, (widgets, mat) in fixes.items():
            node(d, nid)["widgets_values"] = widgets
            if mat is not None:
                link(d, m_pos if mat == "+42" else m_neg, nid, "transform_matrix")
    retitle_groups(d, [
        "Patch 1: yaw +42° — extract & inpaint-encode",
        "Patch 2: yaw −42° — extract & inpaint-encode",
    ])
    set_title(d, 42, "Pinhole 90° → fisheye canvas")
    set_title(d, 67, "Full-frame pass (low-res)")
    add_note(d, (
        "# Outpaint fisheye — SD-inpaint-checkpoint variant\n\n"
        "Same pipeline as `outpainting_fisheye_flux.json`, but the holes are "
        "filled with a classic SD inpainting checkpoint (VAEEncodeForInpaint + "
        "KSampler) instead of Flux: pinhole 90° → 180° fisheye canvas, two ±42° "
        "pinhole patches and a final full-frame pass are inpainted and "
        "composited; RealESRGAN upscales the result.\n\n"
        "- **Set:** input image; positive/negative prompts.\n"
        "- **Requires (models):** `512-inpainting-ema.safetensors`, "
        "`RealESRGAN_x4plus.pth`. No custom node packs.\n"
        "- **Repaired 2026-07:** patch rotations were stored in a pre-2025 "
        "ReprojectImage schema; they are now explicit TransformToMatrix (±42°) "
        "nodes + `inverse` flags. Prefer the Flux variant for quality."
    ))
    save(name, d)


def do_outpainting_fisheye_flux(name="outpainting_fisheye_flux.json"):
    d = load(name)
    repair_widgets(d)
    for nid, inverse in ((42, False), (40, False), (51, True), (80, False),
                         (63, True), (67, False), (72, False)):
        w = node(d, nid)["widgets_values"]
        if len(w) == 8:
            w[6] = inverse  # was stored as 0/45/true mixtures
    retitle_groups(d, [
        "Patch 1: yaw +42° — extract & Flux inpaint",
        "Patch 2: yaw −42° — extract & Flux inpaint",
    ])
    set_title(d, 42, "Pinhole 90° → fisheye canvas")
    set_title(d, 75, "Patch rotation +42°")
    set_title(d, 76, "Patch rotation −42°")
    set_title(d, 67, "Full-frame pass (low-res)")
    set_title(d, 77, "Upscale full pass to 4096")
    add_note(d, (
        "# Outpaint fisheye — Flux variant\n\n"
        "Pinhole 90° image → 180° fisheye canvas; **Flux Inpainting** fills two "
        "90° pinhole patches (rotations ±42° from the TransformToMatrix nodes) "
        "and one low-res full-frame fisheye pass; PorterDuff composites + "
        "RealESRGAN upscale assemble the final 4096² fisheye "
        "(saved as `fluxfish`).\n\n"
        "This is the manual, step-visible version of what "
        "**OutpaintAnyProjection** does in one node — see "
        "`Outpaint_fisheye180.json`.\n\n"
        "- **Set:** input image; prompts in the three Flux Inpainting nodes.\n"
        "- **Requires:** inpainting_flux (auto-installed), FLUX.1-Fill NF4 "
        "weights (first-run download), `RealESRGAN_x4plus.pth`."
    ))
    save(name, d)


def do_sbs180(name="sbs180_workflow.json"):
    d = load(name)
    repair_widgets(d)
    retitle_groups(d, [
        "1. Fisheye metric depth",
        "2. Point cloud & eye-baseline shift",
        "3. Outpaint disocclusions & export equirect",
        "Previews",
    ])
    set_title(d, 33, "Eye baseline (shiftX 0.1)")
    set_title(d, 37, "Right eye (equirect)")
    set_title(d, 44, "Left eye (equirect)")
    add_note(d, (
        "# SBS VR180: synthesize the second eye\n\n"
        "From one 180° fisheye view, synthesizes a stereo pair: fisheye metric "
        "depth → point cloud → clean → shift the camera by the eye baseline "
        "(TransformToMatrix shiftX = 0.1) → re-project to fisheye → four chained "
        "OutpaintAnyProjection passes fill the disocclusions → both eyes are "
        "exported as 180° equirectangular images.\n\n"
        "- **Set:** fisheye input (e.g. from `Outpaint_fisheye180.json`); the "
        "baseline (0.1 ≈ 6.5 cm when depth is metric); prompts optional.\n"
        "- **Requires:** inpainting_flux (auto-installed), Depth-Anything V2.\n"
        "- **Output:** `init_camera_equirect` + `shifted_camera_equirect` — "
        "combine side-by-side for a VR180 player."
    ))
    save(name, d)


def do_pointcloud_inpaint(name="pointcloud_inpaint.json"):
    d = load(name)
    repair_widgets(d)
    set_title(d, 6, "Camera shift (dolly −0.1)")
    set_title(d, 43, "Flux inpaint holes")
    set_title(d, 54, "Align new depth to cloud")
    set_title(d, 55, "Merge old + new points")
    set_groups(d, [
        ("1. Image → point cloud", [1, 46, 18]),
        ("2. Novel view & hole mask", [6, 9, 10, 45, 27, 35, 48, 47]),
        ("3. Flux inpaint", [43, 26]),
        ("4. Lift inpainted region & merge", [49, 54, 50, 55]),
        ("5. Orbit render", [61, 67, 68, 69, 63]),
    ])
    add_note(d, (
        "# Iterative point-cloud inpainting\n\n"
        "Enriches a single-image point cloud with generated content: move the "
        "camera back → render the cloud (holes appear) → grow + invert the "
        "coverage mask → **Flux-inpaint the holes** → re-estimate depth on the "
        "inpainted image → **DepthRenormalizer** aligns it to the original "
        "cloud's depth → lift the new pixels to 3D → **PointCloudUnion** merges "
        "everything → orbit render of the enriched scene.\n\n"
        "One-node alternative: PointcloudTrajectoryEnricher "
        "(`PC_enricher.json`).\n\n"
        "- **Set:** input image; camera shift; inpaint prompt.\n"
        "- **Requires:** inpainting_flux (auto-installed), Depth-Anything V2."
    ))
    save(name, d)


def do_video_camera(name="video_camera.json"):
    d = load(name)
    repair_widgets(d)
    set_groups(d, [
        ("1. Load & pad video", [9, 23, 26, 27, 21, 68]),
        ("2. Metric video depth", [19, 12]),
        ("3. Re-render with new camera", [16, 11, 10, 6]),
        ("4. Masks & composite", [39, 41, 60, 75, 54, 74]),
        ("5. Auto-caption (Florence2)", [63, 62, 61]),
        ("6. WAN VACE re-generation", [29, 30, 37, 32, 33, 34, 28, 31, 36, 35]),
        ("7. Outputs", [4, 13, 14, 38]),
    ])
    set_title(d, 11, "Final camera pose (edit me)")
    set_title(d, 6, "Re-render along trajectory")
    set_title(d, 28, "WAN VACE control")
    add_note(d, (
        "# Re-shoot a video with a new camera move\n\n"
        "Re-renders an input video along a user-defined camera trajectory and "
        "uses WAN 2.1 VACE to regenerate what the new camera reveals:\n\n"
        "1. Video is padded square and depth-estimated per frame "
        "(Video-Depth-Anything metric).\n"
        "2. **VideoCameraMotionSequence** lifts each frame to a point cloud and "
        "re-renders it along the SE(3)-interpolated trajectory.\n"
        "3. Disocclusion masks + the re-rendered frames become VACE "
        "control video/masks; **Florence2** auto-captions the clip as the "
        "prompt; WAN 2.1 VACE 14B fills the gaps.\n\n"
        "- **Set:** video path (VHS Load Video Path); the camera move "
        "(two TransformToMatrix poses); override the auto-caption in "
        "CLIPTextEncode if desired.\n"
        "- **Requires (packs):** VideoHelperSuite, ComfyUI-Florence2, KJNodes "
        "(GetImageRangeFromBatch), ComfyUI-Easy-Use (math), ComfyUI-Image-"
        "Filters (BlurMaskFast).\n"
        "- **Requires (models):** `metric_video_depth_anything_vitl.pth` "
        "(install.sh `depth`), WAN 2.1 VACE 14B + umt5-xxl + WAN VAE "
        "(install.sh `vae`), Florence-2 (auto-download).\n"
        "- **Outputs:** re-rendered composite WEBM, depth WEBM, final VACE clip."
    ))
    save(name, d)


def do_wan_vace_ref(name="wan_vace_ref_to_video.json"):
    d = load(name)
    repair_widgets(d)
    # broken model filename: comma + missing underscore
    n37 = node(d, 37)
    if n37["widgets_values"][0] == "wan2,1_vace14B_fp16.safetensors":
        n37["widgets_values"][0] = "wan2.1_vace_14B_fp16.safetensors"
    set_title(d, 70, "Render control frames + masks")
    set_title(d, 55, "WAN VACE control")
    set_groups(d, [
        ("1. Fisheye image → point cloud", [52, 62, 83, 82, 81, 63, 84]),
        ("2. Camera-motion control video", [77, 70, 78, 80]),
        ("3. WAN VACE generation", [37, 54, 38, 6, 7, 39, 55, 3, 56, 8]),
        ("4. Save", [60, 85, 58]),
    ])
    add_note(d, (
        "# WAN VACE: still image + camera move → video\n\n"
        "Turns a single 180° fisheye still into a camera-move video: metric "
        "depth → point cloud → **CameraMotionNode** renders point-splat frames "
        "and masks along a loaded trajectory; these become the VACE control "
        "video/masks with the original image as the reference, and WAN 2.1 VACE "
        "14B synthesizes the final clip from your prompt.\n\n"
        "- **Set:** fisheye image; trajectory file (record one with "
        "SaveTrajectory); the positive prompt.\n"
        "- **Requires:** WAN 2.1 VACE models (install.sh `vae`), Depth-Anything "
        "V2, VideoHelperSuite (only for the h264/MP4 export — SaveWEBM works "
        "without it).\n"
        "- **Fixed 2026-07:** the UNET filename contained a typo "
        "(`wan2,1_vace14B` → `wan2.1_vace_14B_fp16.safetensors`)."
    ))
    save(name, d)


def main():
    for fn in (
        do_demo_camera, do_outpaint_node_test, do_fisheye_to_pointcloud,
        do_pointcloud, do_pointcloud_walker, do_test_pointcloud_loading,
        do_pc_enricher, do_fisheye_depth, do_outpaint_fisheye180,
        do_outpainting_fisheye_sd, do_outpainting_fisheye_flux, do_sbs180,
        do_pointcloud_inpaint, do_video_camera, do_wan_vace_ref,
    ):
        fn()
        print(f"[ok] {fn.__name__}")
    # video_to_4d_world / video_to_4d_walkable_world already have notes+groups;
    # normalize formatting only.
    for name in ("video_to_4d_world.json", "video_to_4d_walkable_world.json"):
        save(name, load(name))
        print(f"[ok] reformat {name}")


if __name__ == "__main__":
    main()
