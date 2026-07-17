"""World-building nodes: depth-scale anchoring, splat world enrichment along a
trajectory (render -> outpaint -> SHARP -> align -> fuse) and panorama sphere seeding.

Contracts implemented here (see SPEC_4D.md):
  C4: align_depth_scale(new_depth, ref_depth, valid_mask, mode) -> (aligned, scale, shift)

Heavy dependencies (Flux inpainting / diffusers via OutpaintAnyProjection, SHARP)
are only imported/loaded inside methods at call time.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm

try:
    import folder_paths
except ImportError:  # Allow notebook usage outside ComfyUI
    class _FolderPathsStub:
        def __getattr__(self, name):
            raise ModuleNotFoundError(
                "folder_paths is unavailable; this node requires the ComfyUI runtime."
            )

    folder_paths = _FolderPathsStub()

try:
    from . import GS_nodes as _gs
except Exception:
    import GS_nodes as _gs

GaussianSplats = _gs.GaussianSplats
Projection = _gs.Projection
DEVICE_CHOICES = _gs.DEVICE_CHOICES
_resolve_device_choice = _gs._resolve_device_choice
splat_cloud_rotation = _gs.splat_cloud_rotation
_stitch_splats = _gs._stitch_splats

# Zeroth-order real SH constant; rendering with add_sh_bias=True computes
# rgb = C0 * f_dc + 0.5, so seeding uses f_dc = (rgb - 0.5) / C0.
SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Lazy accessors for symbols provided by sibling modules / heavy dependencies
# ---------------------------------------------------------------------------

def _get_render_gaussians():
    """Fetch GS_nodes.render_gaussians (contract C2) with an actionable error."""
    fn = getattr(_gs, "render_gaussians", None)
    if fn is None:
        raise RuntimeError(
            "GS_nodes.render_gaussians is unavailable. Update GS_nodes.py to a version "
            "that provides the module-level render_gaussians function (contract C2)."
        )
    return fn


def _load_outpaint_node_class():
    """Lazy-import OutpaintAnyProjection (pulls in Flux/diffusers machinery)."""
    try:
        from .flux_fisheye_filling_nodes import OutpaintAnyProjection
        return OutpaintAnyProjection
    except Exception:
        pass
    try:
        from flux_fisheye_filling_nodes import OutpaintAnyProjection
        return OutpaintAnyProjection
    except Exception as exc:
        raise RuntimeError(
            "OutpaintAnyProjection could not be imported from flux_fisheye_filling_nodes. "
            "It requires the inpainting_flux custom node package (Flux NF4 inpainting, "
            "diffusers), which this pack's install.py sets up automatically (ComfyUI-Manager "
            "runs it on install). Run install.py or fix custom_nodes/inpainting_flux. "
            f"Import error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# C4: robust depth-scale alignment in the disparity domain
# ---------------------------------------------------------------------------

def align_depth_scale(
    new_depth: torch.Tensor,
    ref_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "scale_shift",
) -> Tuple[torch.Tensor, float, float]:
    """Least-squares scale(+shift) in DISPARITY (1/d) domain on valid_mask pixels,
    robust (clip residual outliers, 2 IRLS rounds). Returns (aligned_depth, scale, shift).

    Fits 1/ref_depth ~= scale * (1/new_depth) + shift over valid pixels and returns
    new_depth remapped through the fitted disparity transform. If the fit is
    degenerate (too few valid pixels, non-positive/non-finite scale), returns the
    input depth unchanged with (scale=1.0, shift=0.0).
    """
    if mode not in ("scale", "scale_shift"):
        raise ValueError(f"Unknown align mode: {mode}")

    nd = torch.as_tensor(new_depth).float()
    # Harmonize devices: the inputs may arrive on different devices (e.g. a
    # CUDA motion mask from MotionMaskFromDepth combined with CPU depth
    # estimates); compute everything on new_depth's device.
    rd = torch.as_tensor(ref_depth).float().to(nd.device)
    vm = torch.as_tensor(valid_mask).float().to(nd.device)

    nd_flat = nd.reshape(-1)
    rd_flat = rd.reshape(-1)
    if vm.numel() == nd_flat.numel():
        vm_flat = vm.reshape(-1)
    else:
        try:
            vm_flat = vm.expand_as(nd).reshape(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"valid_mask shape {tuple(vm.shape)} is not broadcastable to depth shape {tuple(nd.shape)}"
            ) from exc

    eps = 1e-8
    valid = (
        (vm_flat > 0.5)
        & (nd_flat > eps)
        & (rd_flat > eps)
        & torch.isfinite(nd_flat)
        & torch.isfinite(rd_flat)
    )
    if int(valid.sum().item()) < 10:
        return nd.clone(), 1.0, 0.0

    x = 1.0 / nd_flat[valid]  # new disparity
    y = 1.0 / rd_flat[valid]  # reference disparity
    w = torch.ones_like(x)

    scale, shift = 1.0, 0.0
    # Initial weighted LSQ fit + 2 IRLS re-weighting rounds (outlier clipping).
    for _ in range(3):
        sw = w.sum().clamp(min=eps)
        sx = (w * x).sum()
        sy = (w * y).sum()
        if mode == "scale_shift":
            sxx = (w * x * x).sum()
            sxy = (w * x * y).sum()
            denom = sw * sxx - sx * sx
            if float(denom.abs().item()) < eps:
                s = (sxy / sxx.clamp(min=eps)).item()
                b = 0.0
            else:
                s = float(((sw * sxy - sx * sy) / denom).item())
                b = float(((sy - s * sx) / sw).item())
        else:
            sxx = (w * x * x).sum()
            sxy = (w * x * y).sum()
            s = float((sxy / sxx.clamp(min=eps)).item())
            b = 0.0
        scale, shift = s, b

        resid = y - (scale * x + shift)
        sigma = 1.4826 * resid.abs().median()
        sigma = sigma.clamp(min=eps)
        w = (resid.abs() <= 2.5 * sigma).float()
        if float(w.sum().item()) < 10:
            break

    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(shift):
        return nd.clone(), 1.0, 0.0

    disp = scale / nd.clamp(min=eps) + shift
    aligned = 1.0 / disp.clamp(min=eps)
    return aligned, float(scale), float(shift)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_trajectory(trajectory: Any, device: torch.device) -> torch.Tensor:
    """Coerce trajectory input to a [K,4,4] float tensor on device."""
    if isinstance(trajectory, torch.Tensor):
        traj = trajectory
    else:
        traj = torch.as_tensor(trajectory)
    traj = traj.to(device=device, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(f"trajectory must be [K,4,4], got shape {tuple(traj.shape)}")
    return traj


def _project_to_pixels(
    xyz: torch.Tensor,
    projection: str,
    horizontal_fov: float,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-frame points to integer pixel indices.

    Returns (ix [N], iy [N], ray_depth [N], valid [N]) where valid means the point
    is in front of the camera (pinhole) and lands inside the image bounds. Uses the
    same projection math as GS_nodes rendering so pixels line up with renders.
    """
    X, Y, Z = xyz.unbind(-1)
    if projection == "PINHOLE":
        u, v, depth = _gs._xyz_to_pinhole(X, Y, Z, horizontal_fov)
        front = Z > 1e-6
    elif projection == "FISHEYE":
        u, v, depth = _gs._xyz_to_fisheye(X, Y, Z, horizontal_fov)
        front = depth > 1e-6
    else:
        u, v, depth = _gs._xyz_to_equirect(X, Y, Z, horizontal_fov)
        front = depth > 1e-6

    ix = torch.round((u * 0.5 + 0.5) * (width - 1)).long()
    iy = torch.round((v * 0.5 + 0.5) * (height - 1)).long()
    inside = (u >= -1.0) & (u <= 1.0) & (v >= -1.0) & (v <= 1.0)
    valid = front & inside & torch.isfinite(u) & torch.isfinite(v)
    ix = ix.clamp(0, width - 1)
    iy = iy.clamp(0, height - 1)
    return ix, iy, depth, valid


def _pad_f_rest_to_order(splats: GaussianSplats, sh_order: int) -> GaussianSplats:
    """Zero-pad SH coefficients so splats match the requested (higher) SH order.

    Delegates to GS_nodes._pad_sh_order, which handles the renderer's
    channel-major SH layout (cat([f_dc, f_rest]).view(-1, 3, total)) correctly.
    Naively appending zeros to f_rest would shift the green/blue DC terms into
    the red channel's l>=1 slots and corrupt colors.
    """
    return _gs._pad_sh_order(splats, sh_order)


def _match_sh_orders(a: GaussianSplats, b: GaussianSplats) -> Tuple[GaussianSplats, GaussianSplats]:
    """Bring two splat sets to a common (max) SH order via zero padding."""
    return _gs._match_sh_orders(a, b)


def _scale_splats_metric(splats: GaussianSplats, factor: float) -> GaussianSplats:
    """Uniformly rescale splat positions and sizes by a metric factor."""
    out = splats.clone()
    out.xyz = out.xyz * factor
    out.scale = out.scale + math.log(max(factor, 1e-12))
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class DepthScaleAnchor:
    """Aligns a depth map's scale (and optionally shift) to a reference depth map
    using a robust least-squares fit in the disparity domain (contract C4)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "new_depth": ("TENSOR", {"tooltip": "Depth map to be aligned (any shape)."}),
                "ref_depth": ("TENSOR", {"tooltip": "Reference metric depth map (same shape)."}),
                "valid_mask": ("MASK", {"tooltip": "1.0 where both depths are trustworthy."}),
                "mode": (
                    ["scale", "scale_shift"],
                    {"default": "scale_shift", "tooltip": "Fit scale only, or scale + shift, in disparity (1/d) domain."},
                ),
            },
        }

    RETURN_TYPES = ("TENSOR", "FLOAT", "FLOAT")
    RETURN_NAMES = ("aligned_depth", "scale", "shift")
    FUNCTION = "anchor"
    CATEGORY = "Camera/World"
    DESCRIPTION = "Robustly aligns a depth map to a reference depth via disparity-domain scale(+shift)."

    def anchor(
        self,
        new_depth: torch.Tensor,
        ref_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        mode: str = "scale_shift",
    ):
        aligned, scale, shift = align_depth_scale(new_depth, ref_depth, valid_mask, mode=mode)
        return (aligned, scale, shift)


class SplatTrajectoryEnricher:
    """World-expansion loop for Gaussian splats.

    For each pose along a trajectory: render the current splats, detect uncovered
    (hole) regions, fill them with Flux outpainting, lift the filled view to new
    splats with SHARP, align the SHARP metric scale to the rendered reference
    depth, keep only the splats that cover holes, transform them to world space
    and fuse them into the running splat set.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        choices = _gs._list_sharp_checkpoint_choices()
        return {
            "required": {
                "splats": ("GSPLAT",),
                "trajectory": ("TENSOR", {"tooltip": "[K,4,4] world-to-camera matrices of poses to visit."}),
                "camera_projection": (Projection.PROJECTIONS, {}),
                "horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "width": ("INT", {"default": 512, "min": 8, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 8, "max": 8192}),
                "checkpoint": (
                    choices,
                    {
                        "default": _gs._SHARP_DEFAULT_CHECKPOINT_LABEL,
                        "file_chooser": True,
                        "tooltip": "SHARP .pt checkpoint from the input folder, or download the default model.",
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 10, "max": 60}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 30.0}),
                "mask_blur": ("INT", {"default": 5, "min": 0, "max": 512}),
                "hole_min_frac": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001,
                     "tooltip": "Skip a view if the uncovered area is below this fraction of pixels."},
                ),
                "stitch_voxel_size": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0}),
                "max_views": ("INT", {"default": 10, "min": 1, "max": 1000}),
            },
            "optional": {
                "device": (DEVICE_CHOICES, {"default": "auto"}),
                "cache_flux": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Keep the Flux inpainting pipeline loaded between views (avoids a multi-GB "
                                "model reload per view). Disable to free VRAM after each outpaint on "
                                "low-memory GPUs."},
                ),
                "patch_projection": (Projection.PROJECTIONS, {"default": "PINHOLE", "tooltip": "Projection used for the outpaint patch."}),
                "patch_horiz_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 180.0}),
                "patch_res": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "patch_phi": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                "patch_theta": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0}),
            },
        }

    RETURN_TYPES = ("GSPLAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enriched_splats", "last_render", "last_filled")
    FUNCTION = "enrich"
    CATEGORY = "Camera/World"
    DESCRIPTION = (
        "Expands a splat world along a camera trajectory: render, outpaint holes with Flux, "
        "lift with SHARP, scale-align, and smart-stitch the new content."
    )

    @torch.no_grad()
    def enrich(
        self,
        splats: GaussianSplats,
        trajectory: torch.Tensor,
        camera_projection: str,
        horizontal_fov: float,
        width: int,
        height: int,
        checkpoint: str,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        mask_blur: int,
        hole_min_frac: float,
        stitch_voxel_size: float,
        max_views: int,
        device: str = "auto",
        cache_flux: bool = True,
        patch_projection: str = "PINHOLE",
        patch_horiz_fov: float = 90.0,
        patch_res: int = 1024,
        patch_phi: float = 0.0,
        patch_theta: float = 0.0,
    ) -> Tuple[GaussianSplats, torch.Tensor, torch.Tensor]:
        # Fail fast: the SHARP lift (ImageToSplat) is pinhole-only and requires
        # horizontal_fov < 179 degrees. Validating here avoids crashing in the
        # lift step AFTER minutes of rendering + Flux outpainting work.
        if not (0.0 < float(horizontal_fov) < 179.0):
            raise ValueError(
                "SplatTrajectoryEnricher lifts filled views with SHARP (pinhole), which requires "
                f"0 < horizontal_fov < 179 degrees (got {horizontal_fov}). For panoramic worlds "
                "(EQUIRECTANGULAR/FISHEYE with fov >= 179), visit several narrower pinhole poses "
                "along the trajectory instead (e.g. 90-120 degree views after SphereSplatSeed)."
            )
        render_gaussians = _get_render_gaussians()
        outpaint_cls = _load_outpaint_node_class()
        outpaint_node = outpaint_cls()
        image_to_splat = _gs.ImageToSplat()

        target_device = _resolve_device_choice(device)
        current = splats.to(target_device) if splats.xyz.device != target_device else splats
        traj = _coerce_trajectory(trajectory, target_device)

        if camera_projection != "PINHOLE":
            print(
                "[SplatTrajectoryEnricher] Warning: SHARP assumes pinhole geometry; "
                f"lifting filled {camera_projection} views may distort new splats."
            )

        last_render = torch.zeros((1, height, width, 3), device=target_device)
        last_filled = torch.zeros((1, height, width, 3), device=target_device)
        added_views = 0

        for pose in tqdm(traj[: max(1, int(max_views))], desc="Enriching splat world"):
            # 1) Render the current world from this pose.
            image, alpha, disparity = render_gaussians(
                current,
                pose,
                camera_projection,
                horizontal_fov,
                width,
                height,
                max_splats=0,
                opacity_is_logit=True,
                add_sh_bias=True,
                render_mode="auto",
                device=str(target_device).split(":")[0],
            )
            last_render = image

            alpha_map = alpha.view(height, width).to(target_device)
            disp_map = disparity.view(height, width).to(target_device)
            hole_mask = (alpha_map < 0.5).float()

            hole_frac = float(hole_mask.mean().item())
            if hole_frac < hole_min_frac:
                continue

            # 2) Outpaint the uncovered region.
            filled_img, _ = outpaint_node.outpaint_any(
                image,
                input_projection=camera_projection,
                input_horiz_fov=horizontal_fov,
                output_projection=camera_projection,
                output_horiz_fov=horizontal_fov,
                output_width=width,
                output_height=height,
                patch_projection=patch_projection,
                patch_horiz_fov=patch_horiz_fov,
                patch_res=patch_res,
                patch_phi=patch_phi,
                patch_theta=patch_theta,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                # cached=True keeps the Flux NF4 pipeline resident between views
                # (cached=False forced a full multi-GB pipeline reload per view).
                cached=bool(cache_flux),
                guidance_scale=guidance_scale,
                mask_blur=mask_blur,
                mask=hole_mask.unsqueeze(0),
                debug=False,
            )
            last_filled = filled_img

            # 3) Lift the filled view to splats in this camera frame (SHARP, metric).
            new_splats, = image_to_splat.image_to_splat(
                filled_img,
                horizontal_fov,
                checkpoint,
                device,
            )
            new_splats = new_splats.to(target_device)
            if len(new_splats) == 0:
                continue

            # 4) Robust metric-scale alignment against the rendered reference depth.
            #    Reference ray depth from the renderer: disparity = alpha / depth.
            ix, iy, sharp_depth, proj_valid = _project_to_pixels(
                new_splats.xyz, camera_projection, horizontal_fov, width, height
            )
            samp_alpha = alpha_map[iy, ix]
            samp_disp = disp_map[iy, ix]
            overlap = proj_valid & (samp_alpha >= 0.5) & (samp_disp > 1e-6) & (sharp_depth > 1e-6)
            if int(overlap.sum().item()) >= 10:
                d_ref = (samp_alpha[overlap] / samp_disp[overlap]).clamp(min=1e-6)
                ratio = d_ref / sharp_depth[overlap]
                scale_factor = float(ratio.median().item())
                if math.isfinite(scale_factor) and scale_factor > 0.0:
                    new_splats = _scale_splats_metric(new_splats, scale_factor)

            # 5) Keep only NEW content: splats whose projected pixel lies in a hole.
            samp_hole = hole_mask[iy, ix]
            keep = proj_valid & (samp_hole > 0.5)
            if not bool(keep.any().item()):
                continue
            new_splats = new_splats[keep]

            # 6) Camera frame -> world frame (pose is world-to-camera).
            new_world = splat_cloud_rotation(new_splats, torch.inverse(pose))

            # 7) Fuse into the running world. Concatenation is cheap; the full
            #    smart voxel reduce is deferred to a single pass after the loop,
            #    so each view does not re-copy and re-unique-sort the entire
            #    accumulated cloud (O(views x N) work/memory otherwise).
            cur_m, new_m = _match_sh_orders(current, new_world)
            current = _gs._concat_splats([cur_m, new_m])
            added_views += 1

        if added_views > 0 and stitch_voxel_size > 0.0:
            current = _stitch_splats([current], "smart", stitch_voxel_size, 5.0)

        return (current, last_render, last_filled)


class SphereSplatSeed:
    """Seeds a 360-degree splat world from an equirectangular panorama: one Gaussian
    per (subsampled) pixel, placed on a depth sphere around the origin."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular panorama [1,H,W,3]."}),
                "horizontal_fov": ("FLOAT", {"default": 360.0, "min": 1.0, "max": 360.0}),
                "radius": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 10000.0, "tooltip": "Sphere radius used when no depth map is provided."}),
                "splat_scale_frac": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.1, "max": 10.0,
                     "tooltip": "Splat sigma as a fraction of the local point spacing (larger = smoother, fewer holes)."},
                ),
                "stride": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Pixel subsampling stride (1 Gaussian per stride x stride block)."}),
            },
            "optional": {
                "depth": ("TENSOR", {"tooltip": "Optional ray-depth map [H,W] (or [1,H,W]/[H,W,1]) matching the panorama."}),
                "opacity_logit": ("FLOAT", {"default": 6.0, "min": -10.0, "max": 20.0}),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT",)
    RETURN_NAMES = ("splats",)
    FUNCTION = "seed_sphere"
    CATEGORY = "Camera/World"
    DESCRIPTION = "Converts an equirectangular panorama into a Gaussian sphere seeding a 360-degree world."

    @torch.no_grad()
    def seed_sphere(
        self,
        image: torch.Tensor,
        horizontal_fov: float = 360.0,
        radius: float = 5.0,
        splat_scale_frac: float = 1.5,
        stride: int = 2,
        depth: Optional[torch.Tensor] = None,
        opacity_logit: float = 6.0,
        device: str = "auto",
    ) -> Tuple[GaussianSplats]:
        target_device = _resolve_device_choice(device)

        img = image
        if img.dim() == 4:
            img = img[0]
        if img.dim() != 3 or img.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE [1,H,W,3], got shape {tuple(image.shape)}")
        img = img[..., :3].to(device=target_device, dtype=torch.float32)
        H, W = int(img.shape[0]), int(img.shape[1])

        depth_map = None
        if depth is not None:
            d = torch.as_tensor(depth).to(device=target_device, dtype=torch.float32)
            if d.dim() == 3:
                # [1,H,W], [T,H,W] (take first) or [H,W,1]
                d = d[..., 0] if d.shape[-1] == 1 else d[0]
            if d.dim() != 2:
                raise ValueError(f"depth must reduce to [H,W], got shape {tuple(depth.shape)}")
            if d.shape != (H, W):
                d = torch.nn.functional.interpolate(
                    d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True
                )[0, 0]
            depth_map = d.clamp(min=1e-6)

        stride = max(1, int(stride))
        ys = torch.arange(0, H, stride, device=target_device)
        xs = torch.arange(0, W, stride, device=target_device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        # Match the renderer's equirect mapping (GS_nodes._xyz_to_equirect):
        #   u = lon / (fov_rad/2), v = lat / (pi/2), px = (u*0.5+0.5)*(W-1)
        fov_rad = math.radians(horizontal_fov)
        u = xx.float() / max(W - 1, 1) * 2.0 - 1.0
        v = yy.float() / max(H - 1, 1) * 2.0 - 1.0
        lon = u * (fov_rad / 2.0)
        lat = v * (math.pi / 2.0)

        if depth_map is not None:
            d = depth_map[yy, xx]
        else:
            d = torch.full_like(lon, float(radius))

        cos_lat = torch.cos(lat)
        X = d * cos_lat * torch.sin(lon)
        Y = d * torch.sin(lat)
        Z = d * cos_lat * torch.cos(lon)
        xyz = torch.stack([X, Y, Z], dim=-1)

        rgb = img[yy, xx, :]
        # Rendering with add_sh_bias=True evaluates rgb = C0 * f_dc + 0.5.
        f_dc = (rgb - 0.5) / SH_C0

        # Isotropic sigma from local angular spacing (radians per sample) times depth.
        ang_spacing = float(stride) * max(fov_rad / max(W, 1), math.pi / max(H, 1))
        sigma = (splat_scale_frac * ang_spacing * d).clamp(min=1e-6)
        scale = torch.log(sigma).unsqueeze(-1).expand(-1, 3).contiguous()

        n = xyz.shape[0]
        rotation = torch.zeros((n, 4), device=target_device, dtype=torch.float32)
        rotation[:, 0] = 1.0  # identity wxyz quaternion
        opacity = torch.full((n, 1), float(opacity_logit), device=target_device, dtype=torch.float32)
        f_rest = torch.zeros((n, 0), device=target_device, dtype=torch.float32)

        splats = GaussianSplats(
            xyz=xyz,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            f_dc=f_dc,
            f_rest=f_rest,
            sh_order=0,
        )
        return (splats,)


NODE_CLASS_MAPPINGS = {
    "DepthScaleAnchor": DepthScaleAnchor,
    "SplatTrajectoryEnricher": SplatTrajectoryEnricher,
    "SphereSplatSeed": SphereSplatSeed,
}
