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

## Proposed improvements (not applied)

1. **Ship runnable defaults.** Most workflows reference user-local inputs
   (`initial.png`, `Ironcrop.mp4`, `Saved_fisheye_00001_.png`). Bundle one
   small CC0 example image/video in `demo_images/` and point every LoadImage
   at it so a fresh install can hit *Queue* immediately.
2. **Expose workflows in ComfyUI's template browser.** The frontend picks up
   an `example_workflows/` directory inside a custom-node pack. Renaming
   `workflows/` → `example_workflows/` (or copying at publish time via
   `[tool.comfy].includes`) makes every workflow discoverable from the UI
   instead of requiring manual JSON loading.
3. **Trim exotic dependencies from `video_camera.json`.** It pulls five extra
   packs; two are avoidable: `easy mathInt` (Easy-Use) only computes
   `(w−h)/2` for padding — replaceable with core math or fixed values; KJNodes'
   `GetImageRangeFromBatch` (grabs frame 0 for Florence2) is replaceable with
   the built-in `ImageFromBatch` already used in the video_to_4d workflows.
   Also: the `BlurMaskFast` node is a dead end (its output feeds nothing) —
   remove it or wire it into `WanVaceToVideo.control_masks` as presumably
   intended.
4. **Consolidate the fisheye-outpaint family.** Four workflows share one idea
   (`Outpaint_node_test`, `Outpaint_fisheye180`, `outpainting_fisheye_flux`,
   `outpainting_fisheye`). Suggest: keep `Outpaint_fisheye180` as canonical,
   keep the flux variant as the "how it works inside" reference, and move the
   SD-checkpoint variant to `workflows/legacy/` (it needs a 2022-era
   `512-inpainting-ema` checkpoint and produces the weakest results).
5. **Same for depth:** `Fisheye_depth_workflow.json` (45 nodes) is the manual
   expansion of the `FisheyeDepthEstimator` node — the in-canvas note now says
   so; consider `workflows/legacy/` to keep the main folder focused.
6. **Add a trajectory-recording workflow.** `PC_enricher.json` and
   `wan_vace_ref_to_video.json` both *consume* `ComfyUITrajectory_00001.npy`,
   but no workflow *produces* one — a 4-node example
   (`TransformToMatrix ×2 → CameraInterpolationNode → SaveTrajectory`) would
   close the loop.
7. **Retune migrated defaults on the GPU box.** `PointCloudCleaner` (in
   `PointCloud.json`) and the voxel-merge tail of `PointcloudTrajectoryEnricher`
   (in `PC_enricher.json`) were reset to schema defaults; verify visual quality
   and bake in good values.
8. **Guard against future drift in CI.** Add a GitHub Actions job that runs
   `notebooks/validate_workflows.py` (CPU torch) so a node-schema change that
   breaks shipped workflows fails the PR instead of shipping silently.
9. **Version the workflow schema.** Consider adding
   `"extra": {"camera_comfyui_rev": N}` when saving reference workflows, so
   future migrations can target exact revisions instead of sniffing widget
   counts.
