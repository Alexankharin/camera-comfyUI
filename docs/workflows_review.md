# Workflow review — July 2026

Scope: all 17 `workflows/*.json`. Tooling used (kept in `notebooks/`):

* `validate_workflows.py` — checks every stored `widgets_values` array against
  the *current* `INPUT_TYPES` of the node it targets (count + combo values).
  Run it whenever a node's inputs change.
* `rework_workflows_2026_07.py` — the one-shot migration that produced the
  current state of the files (documented below, idempotent-ish).

## What was wrong (now fixed)

ComfyUI applies `widgets_values` **positionally**. When a node gains/loses/
reorders widgets, old workflows load with silently shifted values — no error,
just wrong settings. The validator found 28 such cases across 9 files:

| Node | Old → new widgets | Files affected | Migration applied |
| --- | --- | --- | --- |
| `DepthEstimatorNode` | 2 → 3 (`median_blur_kernel` added) | 5 files, 12 nodes | appended default `1` |
| `FisheyeDepthEstimator` | 6 → 8 (`mode`, `median_blur_kernel` added) | 2 files | inserted `SOFTMERGE` (radius was already configured), appended `1` |
| `PointCloudCleaner` | 2 → 4 (screen-space `width`/`height` added; units changed) | `PointCloud.json` | reset to defaults `[1024, 1024, 1.0, 3]` — old world-unit values were meaningless in the new schema (**retune on GPU box**) |
| `CameraMotionNode` | 6 → 9 (`widen_mask`, `invert_mask`, `points_to_mask`) | 3 files | appended defaults `[0, false, false]` |
| `CameraInterpolationNode` | 0 → 1 (`num_steps`) | 3 files | set `2` (keyframes only — `CameraMotionNode`/`VideoCameraMotionSequence` interpolate frames themselves) |
| `PointcloudTrajectoryEnricher` | 20 → 16 (render/reproject-back options internalized) | `PC_enricher.json` | first 13 kept 1:1, new tail set to defaults |

Beyond widget drift:

* **`outpainting_fisheye.json` was structurally broken**: its seven
  `ReprojectImage` nodes stored patch rotations (±42°) in a pre-2025 widget
  slot that now lands on the `inverse` boolean (42 → truthy). Repaired by
  adding two `TransformToMatrix` nodes (±42°) wired to `transform_matrix`
  inputs and setting proper `inverse` flags — mirroring the structure of
  `outpainting_fisheye_flux.json`.
* **`outpainting_fisheye_flux.json`** stored `45` in one `inverse` slot
  (truthy-by-accident); normalized all seven to real booleans.
* **`wan_vace_ref_to_video.json`** pointed the UNETLoader at
  `wan2,1_vace14B_fp16.safetensors` (comma typo + missing underscore) — no such
  file can exist; fixed to `wan2.1_vace_14B_fp16.safetensors` (matches
  `install.sh`'s download name).
* `Pointcloud walker.json` / `Test pointcloud_loading.json` renamed to
  `pointcloud_walker.json` / `test_pointcloud_loading.json` (spaces break
  shell ergonomics and URL linking).
* README workflow table was stale (claimed `pointcloud_walker` was an "Open3D
  GUI", missed 5 workflows); rewritten with per-workflow extras columns.

Every workflow now carries an embedded **“About this workflow”** MarkdownNote
(purpose, stages, what to set, required packs/models), meaningful group boxes,
and titles on the nodes users are expected to edit.

## Improvements — applied 2026-07-16

Implemented by `notebooks/apply_improvements_2026_07.py` plus repo changes:

1. **Runnable defaults** ✅ — `example_inputs/` ships
   `camera_example_pinhole.jpg`, `camera_example_fisheye.jpg` and
   `ComfyUITrajectory_00001.npy`; `install.py` copies them into ComfyUI's
   `input/` dir (no overwrite), and every LoadImage/LoadTrajectory default
   points at them. Exception: `video_camera.json` still needs a user video
   (none bundled — a clip would bloat the archive).
2. **Template browser** ✅ — `workflows/` is already an accepted template
   directory name (per docs.comfy.org, alongside `example_workflows`), so no
   rename was needed; added the missing same-name `.jpg` thumbnails (10
   workflows, generated 512 px from `demo_images/`) and removed the stray
   `workflows/__init__.py`. Legacy graphs moved to `workflows/legacy/`, which
   keeps them out of the browser.
3. **`video_camera.json` dep trim** ✅ — removed the dead-end `BlurMaskFast`
   (blur radius was 0/0 — a no-op even if wired), the `easy mathInt` pad
   computation and the then-dangling `VHS_VideoInfo`; pad top/bottom now use
   the node's stored values (set to `(W−H)/2`, note explains); swapped
   KJNodes' `GetImageRangeFromBatch` for the built-in `ImageFromBatch`.
   Remaining pack deps: VideoHelperSuite + Florence2 (was five packs).
4. **Fisheye-outpaint consolidation** ✅ — SD-checkpoint variant moved to
   `workflows/legacy/outpainting_fisheye.json`.
5. **Depth consolidation** ✅ — `workflows/legacy/Fisheye_depth_workflow.json`.
6. **Trajectory recorder** ✅ — new `workflows/record_trajectory.json`
   (TransformToMatrix ×2 → CameraInterpolationNode → SaveTrajectory), and the
   bundled example trajectory covers the zero-setup path.
7. **CI guard** ✅ — `.github/workflows/validate.yml` runs the workflow
   validator, installer-logic tests and the 4D smoke suite on CPU torch for
   every PR and push to main.
8. **Schema versioning** ✅ — every workflow now carries
   `extra.camera_comfyui_rev = 1`; bump on the next migration.
9. Bonus: a bidirectional link-integrity sweep found and pruned 8 stale link
   references (pre-existing) plus one dangling `mask` input in
   `outpainting_fisheye_flux.json`.

## Still open

* **Retune migrated defaults on the GPU box.** `PointCloudCleaner` (in
  `PointCloud.json`) and the voxel-merge tail of `PointcloudTrajectoryEnricher`
  (in `PC_enricher.json`) were reset to schema defaults; verify visual quality
  and bake in good values.
* **Load-and-queue pass in real ComfyUI.** All checks here are static; open
  each template once on the GPU box to confirm layout and execution.
* **Bundle a small example video** for `video_camera.json` if archive size
  allows (or document a public sample clip URL in its note).
