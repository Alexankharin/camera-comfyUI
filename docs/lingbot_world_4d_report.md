# LingBot-World 2.0 → 4D video: analysis & integration report

*Research date: 2026-07-13. LingBot-World 2.0 was released 2026-07-09, four days before this report.*

## TL;DR

**LingBot-World 2.0 is not a 3D/4D model — it is a camera-pose- and action-conditioned autoregressive video generator.** It outputs only pixels and maintains no explicit geometry. But it has exactly the property that makes a video-generation model useful for 4D reconstruction: **you command the camera trajectory (poses + intrinsics) of every generated frame**, so every output video is a *posed* video. That turns it into a controllable multi-view video factory whose output can be lifted into 4D Gaussian splats by the existing `video_to_4d_world.json` pipeline in this repo — with the pose-estimation step optionally replaced by the commanded poses.

Feasibility verdicts:

| Question | Verdict |
| --- | --- |
| 4D video from a 3D scene (splat/mesh) | **Yes, indirectly** — render the 3D scene to a seed image, then LingBot animates + explores it. 3D enters only as a rendered start frame; there is no native 3D conditioning. |
| 4D Gaussian-splat video from its output | **Feasible and first-party-endorsed** — the LingBot-World paper itself demonstrates reconstructing its generated videos into point clouds with VGGT-class models, the same VGGT this repo already uses. |
| Drop-in ComfyUI use today | **Not yet** — 14B Wan2.2-based weights, no quantized release for v2, no wrapper support yet ([kijai/WanVideoWrapper#1920](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1920), [Comfy-Org/ComfyUI#12154](https://github.com/Comfy-Org/ComfyUI/issues/12154)); reference inference is 8×GPU `torchrun`. |
| Commercial use | **v2: no** (CC BY-NC-SA 4.0). **v1: yes** (Apache 2.0). This alone may decide which version to build on. |

---

## 1. What LingBot-World 2.0 actually is

**Repos & papers**
- v2 (current): [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2) — "Infinite Worlds with Versatile Interactions", tech report [arXiv:2607.07534](https://arxiv.org/abs/2607.07534), weights [robbyant/lingbot-world-v2-14b-causal-fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast). Released 2026-07-09 by Robbyant (embodied-AI subsidiary of Ant Group).
- v1 (deprecated but still useful): [Robbyant/lingbot-world](https://github.com/robbyant/lingbot-world) — "Advancing Open-source World Models", [arXiv:2601.20540](https://arxiv.org/abs/2601.20540), weights `robbyant/lingbot-world-base-cam` / `-base-act` / `-fast`. Released 2026-01-29.

**Architecture (verified against code + paper)**
- Built on **Wan2.2 i2v-A14B**: a two-expert MoE video diffusion model, ~28B total parameters with **14B active** per denoising step (high-noise expert for global structure, low-noise for detail). Ships the Wan2.1 VAE and umT5-XXL text encoder.
- v2 converts it to **causal, chunk-by-chunk autoregressive generation**: latents are generated `chunk_size` latent frames at a time against a **KV cache** with **sink tokens** and a **local attention window** (`run_fast.sh` uses `--local_attn_size 18 --sink_size 6`). A **MoBA mask** ("Mixture of Bidirectional and Autoregressive Attention Mask") mixes bidirectional attention into teacher forcing to stop the long-horizon quality collapse that plagues autoregressive video. Result: the paper demonstrates an **uninterrupted hour-long session with no perceptible quality decay**.
- Two inference modes: `causal_fast` (distilled few-step; drives **720p @ 60 fps** in their real-time deployment) and `causal_pretrain` (40-step CFG; checkpoint still marked TODO). A single-GPU **1.3B variant is described in the paper but not released**.

**Conditioning inputs — the part that matters for 4D** (from `wan/image2video.py` + `wan/utils/cam_utils.py`)
- **Seed image** (`--image`) + **text prompt**: the world is initialized from one image and a background description. This is the *only* way content enters — no 3D input of any kind.
- **Camera trajectory**: `poses.npy` `[T,4,4]` **camera-to-world, OpenCV convention** + `intrinsics.npy` `[T,4]` = `[fx,fy,cx,cy]`. Converted to per-pixel **Plücker ray embeddings** (`get_plucker_embeddings`), folded into the latent grid and injected per-chunk into the DiT (AdaLN per the tech report). Relative poses are translation-normalized (`compute_relative_poses`), and `interpolate_camera_poses` (SLERP) is provided.
- **Keyboard actions**: `wasd_action.npy` (movement) / `ijkl_action.npy` (view) as multi-hot vectors concatenated onto the Plücker conditioning. v2 adds character actions (attack, archery, spell-cast, shoot, jump, glide) and **chunk-wise text events** (weather, entity spawning, time-of-day), plus a VLM-driven "pilot/director" agentic harness.
- v1 README explicitly recommends **[NVIDIA ViPE](https://github.com/nv-tlabs/vipe)** to extract `poses.npy`/`intrinsics.npy` from an *existing real video* — i.e., the official video→control-signal bridge.

**Inference & hardware**
```bash
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 \
  --frame_num 361 --ckpt_dir lingbot-world-v2-14b-causal-fast \
  --image examples/03/image.jpg --action_path examples/03 \
  --infer_mode causal_fast --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --local_attn_size 18 --sink_size 6
```
- Reference: 8×GPU (FSDP + Ulysses sequence parallel), 480×832, 361 frames (`frame_num` must be 4n+1). Single-GPU runs auto-enable `--offload_model` (T5/DiT swapped to CPU between stages) — expect 80GB-class VRAM for comfortable 14B bf16 inference; there is **no quantized v2 release yet**. v1 has a community **4-bit quant** and `--t5_cpu`, and supports up to 961 frames (~1 min @ 16 fps).
- Requirements: `torch >= 2.4.0`, `flash_attn`.

**License** — v2 code *and* weights are **CC BY-NC-SA 4.0 (non-commercial, share-alike)**; v1 is **Apache 2.0**. Anything commercial built on v2 outputs is off the table; v1 remains the commercially safe option at lower quality/horizon.

---

## 2. Can it turn 3D into 4D video?

**Yes, with the 3D scene entering as a rendered image, not as geometry.** The paper is explicit that the world "is initialized from an initial image and its background description" — there is no splat/mesh/point-cloud conditioning path, and the model "operates without an explicit notion of geometry."

The working recipe, using nodes already in this repo:

1. **Render a seed view** of your static 3D asset: `LoadPlySplat` → `RenderSplat` (or a mesh render) at 832×480+, from a pose with good scene coverage.
2. **Author the camera trajectory you want** in the splat's own coordinate frame (`CameraInterpolationNode` / `CameraTrajectoryNode`), convert to camera-to-world OpenCV `poses.npy` + `intrinsics.npy`.
3. **Feed image + poses + actions/text-events to LingBot-World.** The model animates the scene (wind, characters, weather, spawned entities via text events) while following your camera — i.e., it *invents plausible dynamics* for your static 3D scene. This is "3D → 4D video" in the sense of *generating* the time dimension, not simulating it: physics is learned and imperfect, and the output will drift from your 3D asset's exact geometry the further the camera goes from the seed view.
4. **Optionally lift the result back to 4D splats** (section 3) so the animated version of your scene becomes re-renderable from any camera.

Caveat on fidelity: only the seed frame is constrained by your 3D input. Occluded/unseen regions are hallucinated. For higher fidelity to the source scene you can seed successive generations from renders at multiple poses and stitch — the same strategy `SplatTrajectoryEnricher` already uses with Flux outpainting, but with LingBot providing temporally coherent *video* instead of stills.

---

## 3. Feasibility: 4D Gaussian-splat video from LingBot output

**This is the strongest part of the story.** Three findings, all verified against primary sources:

1. **Posed video for free.** Because generation is conditioned on `poses.npy`/`intrinsics.npy`, every generated frame comes with a commanded camera. A monocular real video gives you poses only after VGGT/COLMAP estimation; LingBot gives you the trajectory you asked for. (Treat commanded poses as *approximate* — the model follows them but is not geometrically exact; see limitations.)
2. **First-party evidence that reconstruction works.** The LingBot-World paper itself demonstrates: *"by leveraging large-scale 3D reconstruction foundation models [lin2025depth, wang2025vggt], we can further convert the generated video sequences into high-quality scene point clouds"*, with point clouds showing *"strong spatial coherence across frames"* (Fig. 16, [arXiv:2601.20540](https://arxiv.org/html/2601.20540v1)). That is literally VGGT — the model behind this repo's `VideoPoseEstimator` — applied to LingBot output by its own authors.
3. **Long-horizon consistency is the v2 headline.** Landmarks stay structurally intact after being out of view for up to ~60 s (v1) and v2 extends coherent generation to hour scale with no perceptible decay. Long consistent orbits are exactly what splat optimization needs.

**How it maps onto known video-to-4D paradigms:**
- **CAT4D-style** ([arXiv:2411.18613](https://arxiv.org/abs/2411.18613)): camera/time-disentangled video diffusion → deformable 3DGS optimization. LingBot is not time-disentangled (you cannot freeze time and move the camera — camera and time advance together in one causal stream), so you *cannot* get true simultaneous multi-view of a dynamic instant from a single run.
- **Monocular 4D lifting** (this repo's pipeline): works on any single posed video — LingBot output qualifies directly and improves on real footage by letting you *choose* a camera path that orbits/parallaxes around the action, which is the single biggest quality lever for monocular 4D reconstruction.
- **Multi-run multi-view**: re-running with the same seed image but different trajectories gives multiple views of the *same static scene* but **different sampled dynamics** (different seeds/action outcomes per run) — usable for static splat fusion, **not** for dynamic 4D supervision. Keep dynamics within one continuous run.

**Bottom line:** treat LingBot-World as a *trajectory-controllable monocular video source* feeding the existing 4D pipeline; don't expect synchronized multi-view rigs out of it.

---

## 4. Concrete pipeline: video → 4D video / 4D splats

### Path A — real video in, 4D world out, LingBot as the world extender

Your existing `video_to_4d_world.json` already handles real-video → 4D. LingBot adds value where that pipeline is weakest: viewpoints the source video never saw.

1. **Base 4D scene from the real video** (existing flow): `VideoPoseEstimator` (VGGT poses/depth) → `ZDepthToRayDepthNode` → `MotionMaskFromDepth` → `VideoToFusedSplats` + `SplatPolish` (static) → `EstimateTracks`/`TracksToTrajectories`/`SplitSplatsByMask`/`BuildSplats4D` (dynamic) → `GSPLAT4D`.
2. **Extract control signals from the same video** with ViPE (officially recommended) or reuse the VGGT poses: `VideoPoseEstimator` outputs world-to-camera `[T,4,4]` → `TrajectoryInvert` → camera-to-world OpenCV → export `poses.npy` + `intrinsics.npy` (VGGT's FOV output gives `fx,fy`; `cx,cy` = image center). *(Small new node needed: `TrajectoryToNpyExport` — trivial, ~20 lines.)*
3. **Continue the world where the video ends**: last real frame = LingBot seed image; author an exploration trajectory (orbit, dolly, walk) continuing from the last real pose; generate 361+ frames.
4. **Lift the generated segment** through the same stage-1 flow and **fuse into the base scene**: `FuseSplats`/`MergeSplats` for statics (scale-anchor with `DepthScaleAnchor` against the base scene's depth), separate `BuildSplats4D` time range for new dynamics. Result: a 4D world larger than the source footage.

### Path B — single image or 3D scene in, 4D splat video out

1. **Seed**: any image, or a render of an existing splat (`RenderSplat`) / mesh.
2. **Trajectory design**: slow orbit or arc around the subject + gentle forward motion — maximize parallax, avoid pure rotation (no baseline → no geometry). Keep FOV fixed; write `poses.npy`/`intrinsics.npy` (c2w, OpenCV; translations get normalized internally, so keep the trajectory scale moderate and re-anchor metric scale later with `DepthScaleAnchor`).
3. **Generate** with `causal_fast`, 480×832, 361 frames; drive dynamics with keyboard/character actions and chunk-wise text events ("a horse gallops through", "rain starts").
4. **Reconstruct** — two pose options:
   - *Trust-but-verify (recommended)*: run `VideoPoseEstimator` on the generated frames anyway; compare with commanded poses (`TrajectoryCompose` of one with `TrajectoryInvert` of the other should be ≈ identity); use VGGT's poses for reconstruction, commanded poses as sanity check. This absorbs the model's camera-following error.
   - *Fast path*: use commanded poses directly, skip VGGT pose estimation, still run its depth head (or `VideoMetricDepthEstimate`) for the depth maps the lifting nodes need.
5. **Lift to 4D**: identical to the existing workflow — motion mask → static fusion (`VideoToFusedSplats` + `SplatPolish`) → tracks (`EstimateTracks` is CoTracker3, works fine on generated footage) → `BuildSplats4D` → `RenderSplats4DVideo` along any novel camera path → `SaveSplats4D`.

### Integration notes for camera-comfyUI

- **Coordinate conventions align well**: LingBot uses OpenCV c2w + `[fx,fy,cx,cy]`, this repo's `TRAJECTORY` is 4×4 matrices with `TrajectoryInvert`/`TrajectoryCompose` already available. Needed glue: (a) `TrajectoryToNpyExport` / `NpyToTrajectory` nodes, (b) optionally a `LingBotGenerate` node wrapping `generate.py` via subprocess for remote/8-GPU boxes — running 14B in-process inside ComfyUI is not realistic today.
- **ComfyUI-native inference isn't there yet**: WanVideoWrapper/ComfyUI support for LingBot checkpoints is an open request blocked on VRAM/quantization ([#1920](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1920), [#12154](https://github.com/Comfy-Org/ComfyUI/issues/12154)). Because it's Wan2.2-architecture, wrapper support and GGUF/FP8 quants are likely to appear quickly; the causal KV-cache/sink/MoBA inference loop is custom, so a naive Wan2.2 loader won't reproduce long-horizon behavior.
- **Pragmatic hardware ladder**: (1) today, single-image experiments on v1 `base-cam` 4-bit quant (Apache 2.0, 480p, camera-pose conditioned — same poses.npy interface) on a 24 GB GPU; (2) v2 14B on a rented 8×A100/H100 node or single 80 GB GPU with offload; (3) wait for the announced 1.3B v2 release for true single-GPU local use.

### Known limitations

- **No geometry inside the model** — all 3D/4D structure comes from post-hoc reconstruction; physics is "imperfect" by the authors' own admission.
- **Camera-following error**: commanded poses ≠ achieved poses exactly (Plücker conditioning is a soft constraint; translations are normalized, so absolute scale is undefined) — always re-anchor scale and consider re-estimating poses.
- **Dynamics are not repeatable across runs** — multi-view supervision of a dynamic instant is impossible; design single continuous runs whose camera moves *around* the action.
- **480×832 native offline resolution** (720p is the real-time streaming mode) — plan on splat-space upscaling or `SplatPolish` against upscaled frames.
- **Generated-content artifacts** (texture shimmer, occasional object morphing) become floaters/ghosts in splat space — the existing `MotionMaskFromDepth` + `DepthEdgeFilter` + `PointCloudCleaner` stack mitigates this, and track-validity filtering in `TracksToTrajectories` matters more than with real footage.
- **License**: v2 is CC BY-NC-SA 4.0 — non-commercial only, share-alike. Use v1 (Apache 2.0) for anything with commercial intent.

---

## Sources

Primary: [lingbot-world-v2 repo](https://github.com/Robbyant/lingbot-world-v2) · [v2 tech report arXiv:2607.07534](https://arxiv.org/abs/2607.07534) · [v2 weights (HF)](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast) · [lingbot-world v1 repo](https://github.com/robbyant/lingbot-world) · [v1 paper arXiv:2601.20540](https://arxiv.org/abs/2601.20540) · [v1 cam weights (HF)](https://huggingface.co/robbyant/lingbot-world-base-cam) · code files `generate.py`, `wan/image2video.py`, `wan/utils/cam_utils.py`, `run_fast.sh` (read directly).
Secondary: [Robbyant press release (2026-07-09)](https://www.businesswire.com/news/home/20260708757367/en/Robbyant-Unveils-LingBot-World-2.0-Pioneering-Hour-Long-Real-Time-Generation-in-World-Models) · [v1 release (2026-01-28)](https://www.businesswire.com/news/home/20260128459962/en/Robbyant-Open-Sources-LingBot-World-a-World-Model-for-Millisecond-Level-Real-Time-Interaction) · [CAT4D arXiv:2411.18613](https://arxiv.org/abs/2411.18613) · [ViPE](https://github.com/nv-tlabs/vipe) · ComfyUI support threads [WanVideoWrapper#1920](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1920), [ComfyUI#12154](https://github.com/Comfy-Org/ComfyUI/issues/12154).

*Method note: claims were gathered by a fan-out research pass (18 sources, 90 raw claims, 25 adversarially verified: 14 confirmed 3-0, 3 refuted, 8 verification-errored) plus direct reading of both repos' inference code and both arXiv papers. The two load-bearing claims whose automated verification errored (v1's video→point-cloud demonstration; the unreleased 1.3B variant) were re-verified manually against the arXiv HTML.*
