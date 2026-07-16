"""Second July-2026 workflow pass: apply the improvements proposed in
docs/workflows_review.md.

1. Point every LoadImage at the bundled example inputs (example_inputs/ is
   copied into ComfyUI's input dir by install.py), so workflows run on a
   fresh install without hunting for files.
2. video_camera.json dependency trim: drop the dead-end BlurMaskFast
   (Image-Filters), the easy-mathInt pad computation (Easy-Use) and the now
   dangling VHS_VideoInfo; swap KJNodes' GetImageRangeFromBatch for the
   built-in ImageFromBatch. Remaining pack deps: VHS + Florence2.
3. Create workflows/record_trajectory.json (TransformToMatrix x2 ->
   CameraInterpolationNode -> SaveTrajectory) - the missing producer for the
   trajectory files PC_enricher / wan_vace_ref_to_video consume.
4. Stamp `extra.camera_comfyui_rev = 1` in every workflow for future
   migrations.

Run: python notebooks/apply_improvements_2026_07.py
"""

import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(REPO, "workflows")

PINHOLE = "camera_example_pinhole.jpg"
FISHEYE = "camera_example_fisheye.jpg"

# workflow file -> example image for its LoadImage node(s)
LOADIMAGE_DEFAULTS = {
    "demo_camera_workflow.json": PINHOLE,
    "Outpaint_node_test.json": PINHOLE,
    "Outpaint_fisheye180.json": PINHOLE,
    "outpainting_fisheye_flux.json": PINHOLE,
    "legacy/outpainting_fisheye.json": PINHOLE,
    "PointCloud.json": PINHOLE,
    "pointcloud_walker.json": PINHOLE,
    "pointcloud_inpaint.json": PINHOLE,
    "fisheye_to_pointcloud.json": FISHEYE,
    "legacy/Fisheye_depth_workflow.json": FISHEYE,
    "PC_enricher.json": FISHEYE,
    "sbs180_workflow.json": FISHEYE,
    "wan_vace_ref_to_video.json": FISHEYE,
}


def load(rel):
    with open(os.path.join(WF, rel), encoding="utf-8") as fh:
        return json.load(fh)


def save(rel, data):
    with open(os.path.join(WF, rel), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def node(data, nid):
    for n in data["nodes"]:
        if n["id"] == nid:
            return n
    raise KeyError(nid)


def set_example_inputs():
    for rel, image in LOADIMAGE_DEFAULTS.items():
        d = load(rel)
        changed = 0
        for n in d["nodes"]:
            if n["type"] == "LoadImage":
                n["widgets_values"][0] = image
                changed += 1
        assert changed, f"{rel}: no LoadImage found"
        save(rel, d)
        print(f"[ok] {rel}: {changed} LoadImage -> {image}")


def remove_node(data, nid):
    """Remove a node and every link touching it; re-index dst slots."""
    n = node(data, nid)
    dead = set()
    for inp in n.get("inputs") or []:
        if inp.get("link") is not None:
            dead.add(inp["link"])
    for out in n.get("outputs") or []:
        dead.update(out.get("links") or [])
    data["nodes"] = [x for x in data["nodes"] if x["id"] != nid]
    data["links"] = [l for l in data["links"] if l[0] not in dead]
    for x in data["nodes"]:
        for inp in x.get("inputs") or []:
            if inp.get("link") in dead:
                inp["link"] = None
        for out in x.get("outputs") or []:
            if out.get("links"):
                out["links"] = [l for l in out["links"] if l not in dead]


def drop_input(data, nid, name):
    """Remove a (widget-converted) input socket and fix slot indices."""
    n = node(data, nid)
    inputs = n.get("inputs") or []
    n["inputs"] = [i for i in inputs if i["name"] != name]
    index = {i["name"]: k for k, i in enumerate(n["inputs"])}
    for l in data["links"]:
        if l[3] == nid:
            # find by link id which input holds it
            for i in n["inputs"]:
                if i.get("link") == l[0]:
                    l[4] = index[i["name"]]


def bbox(data, ids, pad_top=60, pad=20):
    xs, ys, xe, ye = [], [], [], []
    for nid in ids:
        n = node(data, nid)
        p, s = n["pos"], n["size"]
        xs.append(p[0]); ys.append(p[1])
        xe.append(p[0] + s[0]); ye.append(p[1] + s[1])
    x, y = min(xs) - pad, min(ys) - pad_top
    return [x, y, max(xe) + pad - x, max(ye) + pad - y]


def fix_video_camera(rel="video_camera.json"):
    d = load(rel)
    # 1. dead-end mask blur (only Image-Filters usage; radius was 0/0 anyway)
    remove_node(d, 60)
    # 2. pad computation chain (Easy-Use mathInt x2 + VHS_VideoInfo feeding it);
    #    ImagePadForOutpaint falls back to its stored widgets (280/280)
    for nid in (27, 26, 23):
        remove_node(d, nid)
    for name in ("top", "bottom"):
        drop_input(d, 21, name)
    n21 = node(d, 21)
    n21["title"] = "Pad to square — set top/bottom to (W−H)/2"
    # 3. KJNodes GetImageRangeFromBatch -> built-in ImageFromBatch (frame 0)
    n63 = node(d, 63)
    n63["type"] = "ImageFromBatch"
    n63["properties"]["Node name for S&R"] = "ImageFromBatch"
    n63["inputs"][0]["name"] = "image"
    n63["widgets_values"] = [0, 1]  # batch_index, length
    n63["title"] = "First frame (for captioning)"
    # regroup without the removed nodes
    groups = [
        ("1. Load & pad video", [9, 21, 68]),
        ("2. Metric video depth", [19, 12]),
        ("3. Re-render with new camera", [16, 11, 10, 6]),
        ("4. Masks & composite", [39, 41, 75, 54, 74]),
        ("5. Auto-caption (Florence2)", [63, 62, 61]),
        ("6. WAN VACE re-generation", [29, 30, 37, 32, 33, 34, 28, 31, 36, 35]),
        ("7. Outputs", [4, 13, 14, 38]),
    ]
    colors = ["#3f789e", "#a1309b", "#8A8", "#b58b2a", "#88A", "#b06634", "#535"]
    d["groups"] = [{
        "id": i + 1, "title": t, "bounding": bbox(d, ids),
        "color": colors[i % len(colors)], "font_size": 24, "flags": {},
    } for i, (t, ids) in enumerate(groups)]
    # refresh the note (dependency list changed)
    for n in d["nodes"]:
        if n["type"] == "MarkdownNote" and n.get("title") == "About this workflow":
            n["widgets_values"] = [(
                "# Re-shoot a video with a new camera move\n\n"
                "Re-renders an input video along a user-defined camera "
                "trajectory and uses WAN 2.1 VACE to regenerate what the new "
                "camera reveals:\n\n"
                "1. Video is padded square (ImagePadForOutpaint — set "
                "top/bottom to (width−height)/2 for your video) and "
                "depth-estimated per frame (Video-Depth-Anything metric).\n"
                "2. **VideoCameraMotionSequence** lifts each frame to a point "
                "cloud and re-renders it along the SE(3)-interpolated "
                "trajectory.\n"
                "3. Disocclusion masks + re-rendered frames become VACE "
                "control video/masks; **Florence2** auto-captions the clip as "
                "the prompt; WAN 2.1 VACE 14B fills the gaps.\n\n"
                "- **Set:** video path (VHS Load Video Path); the camera move "
                "(two TransformToMatrix poses); pad amounts; override the "
                "auto-caption in CLIPTextEncode if desired.\n"
                "- **Requires (packs):** VideoHelperSuite, ComfyUI-Florence2.\n"
                "- **Requires (models):** `metric_video_depth_anything_vitl"
                ".pth` (install.sh `depth`), WAN 2.1 VACE 14B + umt5-xxl + "
                "WAN VAE (install.sh `vae`), Florence-2 (auto-download).\n"
                "- **Outputs:** re-rendered composite WEBM, depth WEBM, final "
                "VACE clip."
            )]
    save(rel, d)
    print(f"[ok] {rel}: removed BlurMaskFast/mathInt/VideoInfo, "
          f"ImageFromBatch swap, regrouped")


def make_record_trajectory(rel="record_trajectory.json"):
    note = (
        "# Record a camera trajectory\n\n"
        "Produces the `.npy` trajectory file consumed by `PC_enricher.json` "
        "and `wan_vace_ref_to_video.json` (LoadTrajectory): two poses are "
        "SE(3)-interpolated into a smooth 20-step path and saved by "
        "**SaveTrajectory** to your ComfyUI **output** directory.\n\n"
        "- **Set:** the end pose (shift XYZ in scene units — metric if the "
        "cloud came from metric depth — plus theta = pitch, phi = yaw) and "
        "`num_steps`.\n"
        "- **Then:** move the saved file from `output/` to `input/` so "
        "LoadTrajectory can list it. A bundled example "
        "(`ComfyUITrajectory_00001.npy`) is already installed by install.py.\n"
        "- Chain more CameraInterpolationNode segments (or use "
        "CameraTrajectoryNode on a point cloud) for multi-keyframe paths."
    )
    d = {
        "id": "00000000-0000-0000-0000-000000000000",
        "revision": 0,
        "last_node_id": 5,
        "last_link_id": 3,
        "nodes": [
            {
                "id": 1, "type": "TransformToMatrix", "title": "Start pose (identity)",
                "pos": [-500, 320], "size": [315, 154], "flags": {}, "order": 0,
                "mode": 0, "inputs": [],
                "outputs": [{"name": "transformation matrix", "type": "MAT_4X4",
                             "links": [1], "slot_index": 0}],
                "properties": {"Node name for S&R": "TransformToMatrix"},
                "widgets_values": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "id": 2, "type": "TransformToMatrix", "title": "End pose (edit me)",
                "pos": [-500, 540], "size": [315, 154], "flags": {}, "order": 1,
                "mode": 0, "inputs": [],
                "outputs": [{"name": "transformation matrix", "type": "MAT_4X4",
                             "links": [2], "slot_index": 0}],
                "properties": {"Node name for S&R": "TransformToMatrix"},
                "widgets_values": [0.0, 0.0, 0.3, 0.0, 30.0],
            },
            {
                "id": 3, "type": "CameraInterpolationNode",
                "title": "Interpolate 20 poses",
                "pos": [-120, 430], "size": [226, 78], "flags": {}, "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "initial_matrix", "type": "MAT_4X4", "link": 1},
                    {"name": "final_matrix", "type": "MAT_4X4", "link": 2},
                ],
                "outputs": [{"name": "trajectory", "type": "TENSOR",
                             "links": [3], "slot_index": 0}],
                "properties": {"Node name for S&R": "CameraInterpolationNode"},
                "widgets_values": [20],
            },
            {
                "id": 4, "type": "SaveTrajectory", "title": "Save to output/*.npy",
                "pos": [170, 430], "size": [315, 82], "flags": {}, "order": 3,
                "mode": 0,
                "inputs": [{"name": "trajectory", "type": "TENSOR", "link": 3}],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveTrajectory"},
                "widgets_values": ["ComfyUITrajectory"],
            },
            {
                "id": 5, "type": "MarkdownNote", "title": "About this workflow",
                "pos": [-1080, 320], "size": [520, 430], "flags": {}, "order": 4,
                "mode": 0, "inputs": [], "outputs": [], "properties": {},
                "widgets_values": [note], "color": "#432", "bgcolor": "#653",
            },
        ],
        "links": [
            [1, 1, 0, 3, 0, "MAT_4X4"],
            [2, 2, 0, 3, 1, "MAT_4X4"],
            [3, 3, 0, 4, 0, "TENSOR"],
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }
    save(rel, d)
    print(f"[ok] created {rel}")


def stamp_revision():
    for path in glob.glob(os.path.join(WF, "**", "*.json"), recursive=True):
        rel = os.path.relpath(path, WF)
        d = load(rel)
        d.setdefault("extra", {})["camera_comfyui_rev"] = 1
        save(rel, d)
    print("[ok] stamped extra.camera_comfyui_rev = 1")


if __name__ == "__main__":
    set_example_inputs()
    fix_video_camera()
    make_record_trajectory()
    stamp_revision()
