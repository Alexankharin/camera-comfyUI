"""Build workflows/fisheye_static_video_to_4d.json.

Static-fisheye-camera variant of video_to_4d_world.json: because the camera
does not move, no VGGT pose estimation is needed — the trajectory is identity,
per-frame depth comes from the batched FisheyeDepthEstimator, and the whole
static background is a single FisheyeToGaussian prediction of frame 0 split by
the motion mask (outside = static world, inside = dynamic canonical).

Node facts this graph relies on (verified against current INPUT_TYPES):
- FisheyeDepthEstimator is batched: IMAGE [T,H,W,C] -> depthmap [T,H,W,1];
  GS4D nodes squeeze the trailing channel (GS4D_nodes.py::167).
- MotionMaskFromDepth interpolates a [K,4,4] trajectory to T internally.
- TracksToTrajectories' trajectory input is optional and defaults to identity
  ("static camera" per its tooltip) — left unconnected on purpose.
- SplitSplatsByMask returns (inside_splats, outside_splats); its optional
  camera_matrix defaults to identity, which is exactly the frame-0 camera here.

Run: python notebooks/build_fisheye_static_4d.py
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "workflows", "fisheye_static_video_to_4d.json")

NODES = []
LINKS = []
_link_id = 0


def node(nid, ntype, title, pos, size, widgets, inputs=(), outputs=(), mode=0,
         extra=None):
    n = {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
        "flags": {}, "order": len(NODES), "mode": mode,
        "inputs": [dict(i) for i in inputs],
        "outputs": [
            {"name": o[0], "type": o[1], "links": [], "slot_index": k}
            for k, o in enumerate(outputs)
        ],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }
    if title:
        n["title"] = title
    if extra:
        n.update(extra)
    NODES.append(n)
    return n


def link(src, src_slot, dst, dst_input, ltype):
    global _link_id
    _link_id += 1
    sn = next(n for n in NODES if n["id"] == src)
    dn = next(n for n in NODES if n["id"] == dst)
    slot = next(i for i, inp in enumerate(dn["inputs"]) if inp["name"] == dst_input)
    dn["inputs"][slot]["link"] = _link_id
    sn["outputs"][src_slot]["links"].append(_link_id)
    LINKS.append([_link_id, src, src_slot, dst, slot, ltype])


def inp(name, ltype):
    return {"name": name, "type": ltype, "link": None}


NOTE = """# Static fisheye video → 4D Gaussian video

Turns a video shot on a **static 180° fisheye camera** (locked-off shot,
security cam, tripod) into a 4D (3D + time) Gaussian scene, then re-renders it
from a *moving* novel camera.

A static camera needs **no pose estimation** (no VGGT, unlike
`video_to_4d_world.json`): the trajectory is identity, and one frame already
sees the entire static background.

**Stages**
1. **FisheyeDepthEstimator** — per-frame metric *radial* depth on the whole
   fisheye batch (DISTANCE_AWARE multi-view merge).
2. **MotionMaskFromDepth** (identity trajectory) — pixels whose depth changes
   over time are dynamic.
3. **FisheyeToGaussian** on frame 0 → whole-scene splats;
   **SplitSplatsByMask** (FISHEYE 180°) separates the dynamic canonical
   (inside) from the static world (outside).
4. **EstimateTracks** (CoTracker3) + **TracksToTrajectories** (FISHEYE 180°,
   identity poses — its default) lift 2D tracks to 3D control trajectories.
5. **BuildSplats4D** binds the canonical splats to the tracks → 4D scene with
   the static background attached.
6. **RenderSplats4DVideo** renders a novel orbit (pinhole 90°). A second,
   **muted** render replays the original static fisheye view for A/B
   comparison — unmute it (Ctrl+M) to use.

**Set:** the video path + `frame_load_cap` (≤ 64 recommended); the novel-path
end pose; `threshold` in MotionMaskFromDepth (raise it if the mask flickers —
per-frame depth is not temporally consistent).

**Requires:** SHARP + gsplat (auto via install.py), CoTracker3 (torch.hub,
first-use download), Depth-Anything V2 (auto), VideoHelperSuite pack.

**Limitations:** the static world only knows what frame 0 saw — regions
occluded by moving objects at t=0 are holes if the novel camera peeks behind
them; strong depth flicker can leak static pixels into the dynamic set.

**Outputs:** novel-path WEBM, the 4D scene as `.npz` (SaveSplats4D — reload
with LoadSplats4D), optional replay WEBM."""

DA = "Depth-Anything-V2-Metric-Indoor-Base-hf"

# --------------------------------------------------------------------------- #
node(30, "MarkdownNote", "About this workflow", (-2000, 260), (560, 700), [NOTE],
     extra={"color": "#432", "bgcolor": "#653"})

node(1, "VHS_LoadVideoPath", "Load fisheye video (static camera, <= 64 frames)",
     (-1380, 300), (240, 262),
     {
         "video": "input/fisheye_video.mp4",
         "force_rate": 0, "custom_width": 0, "custom_height": 0,
         "frame_load_cap": 49, "skip_first_frames": 0, "select_every_nth": 1,
         "format": "AnimateDiff",
         "videopreview": {"hidden": False, "paused": False, "params": {
             "filename": "input/fisheye_video.mp4", "type": "path",
             "format": "video/mp4", "force_rate": 0, "custom_width": 0,
             "custom_height": 0, "frame_load_cap": 49,
             "skip_first_frames": 0, "select_every_nth": 1}},
     },
     inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[("IMAGE", "IMAGE"), ("frame_count", "INT"),
              ("audio", "AUDIO"), ("video_info", "VHS_VIDEOINFO")])

node(2, "FisheyeDepthEstimator", "Per-frame radial depth (batched)",
     (-1060, 300), (315, 246),
     [DA, 1.0, 90.0, 518, 1024, "DISTANCE_AWARE", 25, 1],
     inputs=[inp("image", "IMAGE")],
     outputs=[("depthmap", "TENSOR"), ("mask", "MASK")])

node(3, "TransformToMatrix", "Static camera (identity pose)",
     (-1060, 640), (315, 154), [0.0, 0.0, 0.0, 0.0, 0.0],
     outputs=[("transformation matrix", "MAT_4X4")])

node(4, "CameraInterpolationNode", "Identity trajectory (interpolated to T)",
     (-680, 680), (226, 78), [2],
     inputs=[inp("initial_matrix", "MAT_4X4"), inp("final_matrix", "MAT_4X4")],
     outputs=[("trajectory", "TENSOR")])

node(5, "MotionMaskFromDepth", "Dynamic-pixel mask (1 = moving)",
     (-680, 300), (315, 202), ["FISHEYE", 180.0, 0.15, 4, 2, "auto"],
     inputs=[inp("depth_seq", "TENSOR"), inp("trajectory", "TENSOR")],
     outputs=[("motion_mask", "MASK")])

node(6, "ImageFromBatch", "Frame 0 (canonical view)",
     (-680, 560), (226, 82), [0, 1],
     inputs=[inp("image", "IMAGE")],
     outputs=[("IMAGE", "IMAGE")])

node(7, "FisheyeToGaussian", "SHARP frame 0 -> whole-scene splats",
     (-300, 460), (330, 290),
     [180.0, 0, 0, "<download default>", "auto", 90.0, 0, "smart", 0.01, 5.0],
     inputs=[inp("image", "IMAGE")],
     outputs=[("splats", "GSPLAT")])

node(8, "EstimateTracks", "CoTracker3 (downloads on first use)",
     (-300, 820), (300, 102), [20, "auto"],
     inputs=[inp("frames", "IMAGE")],
     outputs=[("tracks", "TENSOR"), ("visibility", "TENSOR")])

node(9, "SplitSplatsByMask", "Split: inside = dynamic, outside = static world",
     (100, 300), (315, 174), ["FISHEYE", 180.0, 0.5, "auto"],
     inputs=[inp("splats", "GSPLAT"), inp("mask", "MASK"),
             inp("camera_matrix", "MAT_4X4")],
     outputs=[("inside_splats", "GSPLAT"), ("outside_splats", "GSPLAT")])

node(10, "TracksToTrajectories", "Lift tracks to 3D (identity poses = default)",
     (100, 700), (315, 190), ["FISHEYE", 180.0, 0.5, "auto"],
     inputs=[inp("tracks", "TENSOR"), inp("visibility", "TENSOR"),
             inp("depth_seq", "TENSOR"), inp("trajectory", "TENSOR")],
     outputs=[("trajectories3d", "TENSOR"), ("track_valid", "TENSOR")])

node(11, "BuildSplats4D", "Bind canonical to tracks -> 4D scene",
     (500, 440), (315, 190), [0, 4, 0.0, "auto"],
     inputs=[inp("canonical", "GSPLAT"), inp("trajectories3d", "TENSOR"),
             inp("static", "GSPLAT"), inp("times", "TENSOR"),
             inp("track_valid", "TENSOR")],
     outputs=[("splats4d", "GSPLAT4D")])

node(12, "TransformToMatrix", "Novel path: start (original camera)",
     (500, 720), (315, 154), [0.0, 0.0, 0.0, 0.0, 0.0],
     outputs=[("transformation matrix", "MAT_4X4")])

node(13, "TransformToMatrix", "Novel path: end (small orbit — edit me)",
     (500, 920), (315, 154), [0.15, 0.0, 0.1, 0.0, -10.0],
     outputs=[("transformation matrix", "MAT_4X4")])

node(14, "CameraInterpolationNode", "Novel camera path",
     (880, 820), (226, 78), [2],
     inputs=[inp("initial_matrix", "MAT_4X4"), inp("final_matrix", "MAT_4X4")],
     outputs=[("trajectory", "TENSOR")])

node(15, "RenderSplats4DVideo", "Render 4D along novel path (pinhole 90°)",
     (900, 300), (315, 266), [49, 0.0, 1.0, "PINHOLE", 90.0, 768, 768, "auto", 0, "auto"],
     inputs=[inp("splats4d", "GSPLAT4D"), inp("trajectory", "TENSOR")],
     outputs=[("images", "IMAGE"), ("masks", "MASK"), ("disparity", "TENSOR")])

node(16, "RenderSplats4DVideo", "MUTED: replay original fisheye view (A/B check)",
     (900, 1060), (315, 266), [49, 0.0, 1.0, "FISHEYE", 180.0, 1024, 1024, "auto", 0, "auto"],
     inputs=[inp("splats4d", "GSPLAT4D"), inp("trajectory", "TENSOR")],
     outputs=[("images", "IMAGE"), ("masks", "MASK"), ("disparity", "TENSOR")],
     mode=2)

node(17, "SaveWEBM", None, (1300, 300), (315, 437),
     ["4d_fisheye_novel", "vp9", 24, 32],
     inputs=[inp("images", "IMAGE")])

node(18, "SaveSplats4D", "Save 4D scene (.npz)", (1300, 800), (315, 106),
     ["ComfyUISplat4D_fisheye", False],
     inputs=[inp("splats4d", "GSPLAT4D")])

node(19, "SaveWEBM", "MUTED: replay output", (1300, 1060), (315, 437),
     ["4d_fisheye_replay", "vp9", 24, 32],
     inputs=[inp("images", "IMAGE")], mode=2)

# --------------------------------------------------------------------------- #
link(1, 0, 2, "image", "IMAGE")
link(1, 0, 6, "image", "IMAGE")
link(1, 0, 8, "frames", "IMAGE")
link(2, 0, 5, "depth_seq", "TENSOR")
link(2, 0, 10, "depth_seq", "TENSOR")
link(3, 0, 4, "initial_matrix", "MAT_4X4")
link(3, 0, 4, "final_matrix", "MAT_4X4")
link(4, 0, 5, "trajectory", "TENSOR")
link(4, 0, 16, "trajectory", "TENSOR")
link(5, 0, 9, "mask", "MASK")
link(6, 0, 7, "image", "IMAGE")
link(7, 0, 9, "splats", "GSPLAT")
link(8, 0, 10, "tracks", "TENSOR")
link(8, 1, 10, "visibility", "TENSOR")
link(9, 0, 11, "canonical", "GSPLAT")
link(9, 1, 11, "static", "GSPLAT")
link(10, 0, 11, "trajectories3d", "TENSOR")
link(10, 1, 11, "track_valid", "TENSOR")
link(11, 0, 15, "splats4d", "GSPLAT4D")
link(11, 0, 16, "splats4d", "GSPLAT4D")
link(11, 0, 18, "splats4d", "GSPLAT4D")
link(12, 0, 14, "initial_matrix", "MAT_4X4")
link(13, 0, 14, "final_matrix", "MAT_4X4")
link(14, 0, 15, "trajectory", "TENSOR")
link(15, 0, 17, "images", "IMAGE")
link(16, 0, 19, "images", "IMAGE")
# NOTE: TracksToTrajectories.trajectory stays unconnected on purpose — its
# default is identity poses, i.e. exactly the static camera.


def bbox(ids, pad_top=60, pad=20):
    ns = [n for n in NODES if n["id"] in ids]
    x = min(n["pos"][0] for n in ns) - pad
    y = min(n["pos"][1] for n in ns) - pad_top
    x2 = max(n["pos"][0] + n["size"][0] for n in ns) + pad
    y2 = max(n["pos"][1] + n["size"][1] for n in ns) + pad
    return [x, y, x2 - x, y2 - y]


GROUPS = [
    ("1. Load fisheye video", [1]),
    ("2. Per-frame fisheye depth", [2]),
    ("3. Static camera trajectory", [3, 4]),
    ("4. Motion mask", [5]),
    ("5. Frame-0 splats & static/dynamic split", [6, 7, 9]),
    ("6. Dynamic tracks -> 3D", [8, 10]),
    ("7. 4D scene", [11, 18]),
    ("8. Render novel path (+ muted replay)", [12, 13, 14, 15, 16, 17, 19]),
]
COLORS = ["#3f789e", "#a1309b", "#8A8", "#b58b2a", "#88A", "#b06634", "#535",
          "#3f789e"]

workflow = {
    "id": "00000000-0000-0000-0000-000000000000",
    "revision": 0,
    "last_node_id": 30,
    "last_link_id": _link_id,
    "nodes": NODES,
    "links": LINKS,
    "groups": [{
        "id": i + 1, "title": t, "bounding": bbox(ids),
        "color": COLORS[i % len(COLORS)], "font_size": 24, "flags": {},
    } for i, (t, ids) in enumerate(GROUPS)],
    "config": {},
    "extra": {"camera_comfyui_rev": 1},
    "version": 0.4,
}

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(workflow, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {OUT}: {len(NODES)} nodes, {len(LINKS)} links")
