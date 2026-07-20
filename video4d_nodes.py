"""Per-frame 4D splat video nodes.

An alternative "video gaussian" pipeline for footage from a static camera:

  1. Temporally-consistent metric depth from a *video* model
     (`VideoMetricDepthEstimate`, Video-Depth-Anything) — this is what keeps
     the static geometry stable across frames.
  2. `TemporalStaticPlate` — robust static/dynamic split from temporal
     median statistics of color AND depth (photometric changes catch motion
     that depth-warp residuals miss).
  3. `SplatFramesFromVideo` — every frame becomes its own splat cloud:
     a shared static base (built once from the median plate, identical in
     every frame) plus per-frame dynamic splats inside the union of all
     motion masks. Pixel-identity ordering gives the constant splat count +
     per-index correspondence that temporal splat compressors expect.
  4. `RenderSplatSequence` — preview: render each frame's splats along an
     (optionally interpolated) camera trajectory.
  5. `SaveSplats4DVideo` — export antimatter15 .splat frames and encode them
     into one streamable .splat4d file (https://github.com/adamraudonis/splats4D).
"""

import math
import os
import shutil
import subprocess
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

from .pointcloud_nodes import (
    Projection,
    interpolate_se3,
    pinhole_depth_to_XYZ,
    fisheye_depth_to_XYZ,
    equirect_depth_to_XYZ,
)
from .GS_nodes import GaussianSplats, RenderSplat

C0 = 0.28209479177387814

_XYZ_FN = {
    "PINHOLE": pinhole_depth_to_XYZ,
    "FISHEYE": fisheye_depth_to_XYZ,
    "EQUIRECTANGULAR": equirect_depth_to_XYZ,
}


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_thw(t: torch.Tensor) -> torch.Tensor:
    """[T,H,W,1] / [T,H,W] / [H,W] → [T,H,W]."""
    if t.dim() == 4:
        t = t.squeeze(-1)
    elif t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of [T,H,W] float mask by `radius` pixels."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask.unsqueeze(1), k, stride=1, padding=radius).squeeze(1)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return 1.0 - _dilate(1.0 - mask, radius)


def _unit_directions(projection: str, fov: float, height: int, width: int,
                     device: torch.device) -> torch.Tensor:
    """Per-pixel unit ray directions [H,W,3] for the pack's projections
    (depth in this pack is radial along these rays)."""
    ones = torch.ones(height, width, device=device)
    X, Y, Z = _XYZ_FN[projection](ones, fov)
    dirs = torch.stack((X, Y, Z), dim=-1)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _circular_mask(projection: str, height: int, width: int,
                   device: torch.device) -> torch.Tensor:
    """Valid-pixel mask [H,W]: the image circle for fisheye, ones otherwise."""
    if projection != "FISHEYE":
        return torch.ones(height, width, device=device)
    u = torch.linspace(-1.0, 1.0, width, device=device).unsqueeze(0).expand(height, width)
    v = torch.linspace(-1.0, 1.0, height, device=device).unsqueeze(1).expand(height, width)
    return ((u * u + v * v) <= 1.0).float()


class TemporalStaticPlate:
    """
    Static/dynamic decomposition for a locked-off camera from temporal
    statistics. Per-pixel temporal medians of color and (log) depth define a
    static plate; a pixel is dynamic in frame t when its color OR depth
    deviates from the plate. Combining both signals catches low-texture
    motion (depth) and depth-flicker-immune motion (color) — much more
    robust than depth-warp residuals alone.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "[T,H,W,3] video frames (static camera)"}),
                "depth_seq": ("TENSOR", {"tooltip": "[T,H,W] temporally consistent depth (VideoMetricDepthEstimate)"}),
                "color_threshold": ("FLOAT", {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                                              "tooltip": "Max |RGB - median| (0-1) still considered static"}),
                "depth_threshold": ("FLOAT", {"default": 0.10, "min": 0.005, "max": 2.0, "step": 0.005,
                                              "tooltip": "Max |log depth - log median| still considered static (~relative depth change)"}),
                "open_radius": ("INT", {"default": 2, "min": 0, "max": 16,
                                        "tooltip": "Morphological opening radius — kills isolated speckles"}),
                "close_radius": ("INT", {"default": 16, "min": 0, "max": 128,
                                         "tooltip": "Morphological closing radius (pixels — scale with resolution) — fills the interior of moving objects whose center matches the median (low-texture bodies)"}),
                "union_close_radius": ("INT", {"default": 48, "min": 0, "max": 256,
                                               "tooltip": "Stronger closing on the UNION mask: the union is outlined by the mover's full sweep, so a big radius fills its whole footprint (low-motion torso included)"}),
                "dilate": ("INT", {"default": 6, "min": 0, "max": 64,
                                   "tooltip": "Safety margin grown around detected motion"}),
                "temporal_dilate": ("INT", {"default": 1, "min": 0, "max": 8,
                                            "tooltip": "A pixel dynamic in frame t is also dynamic in t±k"}),
                "bg_percentile": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "Background completion inside moving regions: per-pixel depth quantile treated as the revealed background (moving objects are closer than what they occlude)"}),
                "bg_min_separation": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01,
                                                "tooltip": "Background counts as REVEALED only if its depth quantile is this much beyond the median (log depth). A mover swaying in place fakes small separations — those pixels go to unobserved_mask for generative fill instead."}),
            },
            "optional": {
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TENSOR", "MASK", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("static_rgb", "static_depth", "masks", "union_mask", "mask_preview", "unobserved_mask")
    FUNCTION = "build_plate"
    CATEGORY = "Camera/Video4D"
    DESCRIPTION = "Median static plate + per-frame dynamic masks from color & depth deviation (static camera). unobserved_mask marks background never revealed by motion — feed it to a generative filler (e.g. WAN VACE via BlendImagesByMask)."

    def build_plate(
        self,
        frames: torch.Tensor,
        depth_seq: torch.Tensor,
        color_threshold: float,
        depth_threshold: float,
        open_radius: int,
        close_radius: int,
        union_close_radius: int,
        dilate: int,
        temporal_dilate: int,
        bg_percentile: float,
        bg_min_separation: float = 0.25,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = _pick_device(device)
        rgb = frames.to(dev).float()                      # [T,H,W,3]
        depth = _to_thw(depth_seq).to(dev).float().clamp(min=1e-6)  # [T,H,W]
        T = rgb.shape[0]

        med_rgb = rgb.median(dim=0).values                # [H,W,3]
        log_d = depth.log()
        med_logd = log_d.median(dim=0).values             # [H,W]

        rgb_dev = (rgb - med_rgb).abs().amax(dim=-1)      # [T,H,W]
        d_dev = (log_d - med_logd).abs()                  # [T,H,W]
        dynamic = ((rgb_dev > color_threshold) | (d_dev > depth_threshold)).float()

        # spatial cleanup: opening kills speckle; closing fills object
        # interiors (a body whose center pixels match the median — e.g.
        # low-texture skin — is detected only at its outline); then grow a
        # safety margin
        dynamic = _dilate(_erode(dynamic, open_radius), open_radius)
        dynamic = _erode(_dilate(dynamic, close_radius), close_radius)
        dynamic = _dilate(dynamic, dilate)

        # temporal dilation: motion in t also flags t±k (catches boundaries
        # and single-frame detection dropouts)
        if temporal_dilate > 0:
            padded = torch.cat(
                [dynamic[:1].expand(temporal_dilate, -1, -1), dynamic,
                 dynamic[-1:].expand(temporal_dilate, -1, -1)], dim=0)
            dynamic = padded.unfold(0, 2 * temporal_dilate + 1, 1).amax(dim=-1)

        union = dynamic.amax(dim=0, keepdim=True)         # [1,H,W]
        union = _erode(_dilate(union, max(close_radius, union_close_radius)),
                       max(close_radius, union_close_radius))

        # Background completion inside the union: a moving object is closer
        # than what it occludes, so the deep-depth quantile over time is the
        # revealed background. Color = robust mean over the frames that
        # actually observed that background (single-frame picks are speckly).
        bg_logd = torch.quantile(log_d, bg_percentile, dim=0)     # [H,W]
        w = (log_d >= bg_logd - 0.05).float().unsqueeze(-1)       # [T,H,W,1]
        bg_rgb = (rgb * w).sum(dim=0) / w.sum(dim=0).clamp(min=1e-6)

        u = union[0].bool()
        plate_rgb = torch.where(u.unsqueeze(-1), bg_rgb, med_rgb)
        plate_logd = torch.where(u, bg_logd, med_logd)

        # Never-revealed background (the mover never leaves those pixels →
        # bg quantile stays at the mover's own depth): treat as holes and
        # fill by diffusion from surrounding observed background, so novel
        # views show plausible wall/floor instead of black voids.
        unobserved = u & ((bg_logd - med_logd) < bg_min_separation)

        # Hair/mixed fringes of the mover leak into the median plate just
        # OUTSIDE the union (they were below the motion thresholds): plate
        # pixels sitting on a depth edge near the union are refilled as
        # background too, so the static layer is clean under the seam.
        H, W = med_logd.shape
        gpx = max(1, W // 1024)
        k = 4 * gpx + 1
        sm = F.avg_pool2d(plate_logd.unsqueeze(0).unsqueeze(0), k,
                          stride=1, padding=k // 2)[0, 0]
        gx2 = F.pad((sm[:, gpx:] - sm[:, :-gpx]).abs(), (0, gpx))
        gy2 = F.pad((sm[gpx:, :] - sm[:-gpx, :]).abs(), (0, 0, 0, gpx))
        plate_edge = torch.maximum(gx2, gy2) > 2.0 * depth_threshold
        near_union = _dilate(union, 24 * gpx)[0] > 0.5
        unobserved = unobserved | (plate_edge & near_union)

        unobserved_out = unobserved.float().unsqueeze(0)          # [1,H,W]
        if unobserved.any():
            known = (~unobserved).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            feat = torch.cat([plate_rgb.permute(2, 0, 1),
                              plate_logd.unsqueeze(0)]).unsqueeze(0)  # [1,4,H,W]
            feat = feat * known
            # depth fills by MAX-propagation (farthest neighbor): the hidden
            # surface is behind the mover, so averaging near+far neighbors
            # would park fill splats mid-air in front of it.
            LOGD_FLOOR = -20.0
            depth_fill = torch.where(known > 0.5, feat[:, 3:4],
                                     torch.full_like(feat[:, 3:4], LOGD_FLOOR))
            for _ in range(256):
                blur_f = F.avg_pool2d(feat, 9, stride=1, padding=4)
                blur_k = F.avg_pool2d(known, 9, stride=1, padding=4)
                fill = blur_f / blur_k.clamp(min=1e-6)
                depth_prop = F.max_pool2d(depth_fill, 9, stride=1, padding=4)
                newly = (blur_k > 1e-6) & (known < 0.5)
                feat = torch.where(newly.expand_as(feat), fill, feat)
                depth_fill = torch.where(newly, depth_prop, depth_fill)
                known = torch.where(newly, torch.ones_like(known), known)
                if bool((known > 0.5).all()):
                    break
            filled = feat[0]
            m = unobserved.unsqueeze(-1)
            plate_rgb = torch.where(m, filled[:3].permute(1, 2, 0), plate_rgb)
            # Never-observed background must sit clearly BEHIND the mover —
            # the hole's neighbors are mostly the mover itself, so propagated
            # depth alone would park the fill at the mover's depth and smear
            # over it from novel views.
            behind = med_logd + 0.35
            plate_logd = torch.where(
                unobserved, torch.maximum(depth_fill[0, 0], behind), plate_logd)

        # previews: motion painted red on the frames
        overlay = rgb.clone()
        m = dynamic.unsqueeze(-1)
        overlay = overlay * (1 - 0.5 * m) + 0.5 * m * torch.tensor(
            [1.0, 0.0, 0.0], device=dev)

        static_rgb = plate_rgb.unsqueeze(0)               # [1,H,W,3]
        static_depth = plate_logd.exp().unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
        return (static_rgb.cpu(), static_depth.cpu(), dynamic.cpu(),
                union.cpu(), overlay.cpu(), unobserved_out.cpu())


class SplatFramesFromVideo:
    """
    Build one splat cloud per frame with pixel-identity correspondence:
    static splats come from the median plate (identical in every frame,
    listed first); dynamic splats live at the pixels of the UNION of all
    motion masks and take their depth/color from the current frame. Total
    count and per-index meaning are constant across frames — exactly what
    temporal splat compressors (.splat4d) exploit.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frames": ("IMAGE",),
                "depth_seq": ("TENSOR",),
                "static_rgb": ("IMAGE",),
                "static_depth": ("TENSOR",),
                "union_mask": ("MASK",),
                "projection": (Projection.PROJECTIONS, {"default": "FISHEYE"}),
                "horizontal_fov": ("FLOAT", {"default": 180.0, "min": 1.0, "max": 360.0}),
                "stride": ("INT", {"default": 2, "min": 1, "max": 8,
                                   "tooltip": "Pixel subsampling; 2 → quarter the splats"}),
                "splat_scale": ("FLOAT", {"default": 1.3, "min": 0.05, "max": 10.0, "step": 0.05,
                                          "tooltip": "Splat radius multiplier (× pixel footprint)"}),
                "opacity": ("FLOAT", {"default": 0.95, "min": 0.05, "max": 0.995, "step": 0.005}),
                "max_depth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0,
                                        "tooltip": "Clamp depth (0 = off)"}),
                "edge_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                                             "tooltip": "Suppress 'flying pixel' splats at depth discontinuities: |Δ log depth| per pixel above this is an edge (0 = off). Static edge splats are dropped (pushed to background near the motion seam); dynamic ones are made transparent per frame (count stays constant)."}),
                "snap_depth_edges": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Bimodal depth snap in silhouette bands: mixed hair/background pixels get the nearer of the local foreground/background depth instead of a floating in-between value — kills 'hair veil' smears under camera motion."}),
                "seam_band": ("INT", {"default": 16, "min": 0, "max": 128,
                                      "tooltip": "Half-width (px) of the band around the union-mask boundary that is densified to stride 1 and where static edge splats are pushed behind instead of dropped — closes the seam holes between static and dynamic layers."}),
            },
            "optional": {
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("GSPLAT_SEQ", "GSPLAT", "INT", "INT")
    RETURN_NAMES = ("splat_frames", "static_splats", "static_count", "dynamic_count")
    FUNCTION = "build_frames"
    CATEGORY = "Camera/Video4D"
    DESCRIPTION = "Per-frame splats: shared static base + per-frame dynamic pixels (constant count & order)."

    def build_frames(
        self,
        frames: torch.Tensor,
        depth_seq: torch.Tensor,
        static_rgb: torch.Tensor,
        static_depth: torch.Tensor,
        union_mask: torch.Tensor,
        projection: str,
        horizontal_fov: float,
        stride: int,
        splat_scale: float,
        opacity: float,
        max_depth: float,
        edge_threshold: float,
        snap_depth_edges: bool = True,
        seam_band: int = 16,
        device: str = "auto",
    ) -> Tuple[List[GaussianSplats], GaussianSplats, int, int]:
        dev = _pick_device(device)
        rgb = frames.to(dev).float()                              # [T,H,W,3]
        depth = _to_thw(depth_seq).to(dev).float().clamp(min=1e-6)
        plate_rgb = static_rgb.to(dev).float()[0]                 # [H,W,3]
        plate_d = _to_thw(static_depth).to(dev).float()[0].clamp(min=1e-6)
        union = _to_thw(union_mask).to(dev).float()[0] > 0.5      # [H,W]
        T, H, W = depth.shape
        if max_depth > 0:
            depth = depth.clamp(max=max_depth)
            plate_d = plate_d.clamp(max=max_depth)

        dirs = _unit_directions(projection, horizontal_fov, H, W, dev)  # [H,W,3]
        valid = _circular_mask(projection, H, W, dev) > 0.5

        gpx = max(1, W // 1024)            # resolution-relative pixel step

        def smooth_log(d: torch.Tensor) -> torch.Tensor:
            k = 4 * gpx + 1
            ld = d.clamp(min=1e-6).log().unsqueeze(0).unsqueeze(0)
            return F.avg_pool2d(ld, k, stride=1, padding=k // 2)[0, 0]

        def edge_map(d: torch.Tensor, thr: Optional[float] = None) -> torch.Tensor:
            """[H,W] depth → bool edge mask via |Δ log depth| (dilated by stride).

            The log depth is pre-smoothed so only real silhouette jumps count,
            not estimator noise or upsampling seams. Smoothing kernel and
            gradient baseline scale with resolution so the threshold means the
            same thing at 1024 and 4096."""
            ld = smooth_log(d)
            gx = F.pad((ld[:, gpx:] - ld[:, :-gpx]).abs(), (0, gpx))
            gy = F.pad((ld[gpx:, :] - ld[:-gpx, :]).abs(), (0, 0, 0, gpx))
            e = torch.maximum(gx, gy) > (edge_threshold if thr is None else thr)
            return _dilate(e.float().unsqueeze(0), stride).squeeze(0) > 0.5

        def bimodal_snap(d: torch.Tensor) -> torch.Tensor:
            """Snap silhouette-band pixels to the nearer of the local
            foreground/background depth. Soft boundaries (hair) otherwise
            produce splats at floating in-between depths that fan out into
            'smoke veil' streaks under camera parallax."""
            ld = d.clamp(min=1e-6).log()
            band = edge_map(d, 0.5 * edge_threshold)  # includes soft ramps
            band = _dilate(band.float().unsqueeze(0), 2 * gpx)[0] > 0.5
            r = 4 * gpx
            kk = 2 * r + 1
            x = ld.unsqueeze(0).unsqueeze(0)
            bg = F.max_pool2d(x, kk, stride=1, padding=r)[0, 0]
            fg = -F.max_pool2d(-x, kk, stride=1, padding=r)[0, 0]
            snapped = torch.where((ld - fg) <= (bg - ld), fg, bg)
            return torch.where(band, snapped, ld).exp()

        if snap_depth_edges and edge_threshold > 0:
            plate_d = bimodal_snap(plate_d)
            depth = torch.stack([bimodal_snap(depth[t]) for t in range(T)])

        # subsampling grid — densified to stride 1 in a band around the
        # union boundary, where the static/dynamic seam otherwise shows
        # 'comb teeth' gaps at glancing parallax
        u1 = union.float().unsqueeze(0)
        seam = torch.zeros(H, W, dtype=torch.bool, device=dev)
        if seam_band > 0:
            seam = (_dilate(u1, seam_band)[0] > 0.5) & ~(_erode(u1, seam_band)[0] > 0.5)
        keep = torch.zeros(H, W, dtype=torch.bool, device=dev)
        keep[::stride, ::stride] = True
        keep = keep | seam
        valid = valid & keep
        stride_map = torch.full((H, W), float(stride), device=dev)
        stride_map[seam] = 1.0

        # The static layer covers ALL valid pixels: outside the union it is
        # the median plate, inside it the completed background — so novel
        # views see background instead of holes behind moving objects. The
        # dynamic layer draws on top of it per frame.
        static_valid = valid
        if edge_threshold > 0:
            e_static = edge_map(plate_d)
            near_union = _dilate(u1, max(seam_band, 1))[0] > 0.5
            # near the motion seam: push edge splats to the local background
            # (keeps coverage under the seam); elsewhere: drop them
            push = e_static & near_union
            static_valid = static_valid & ~(e_static & ~near_union)
            if push.any():
                ldp = plate_d.clamp(min=1e-6).log()
                r = 4 * gpx
                bgp = F.max_pool2d(ldp.unsqueeze(0).unsqueeze(0), 2 * r + 1,
                                   stride=1, padding=r)[0, 0]
                plate_d = torch.where(push, bgp.exp(), plate_d)
        static_idx = static_valid.nonzero(as_tuple=False)         # [Ns,2] (y,x)
        dyn_idx = (valid & union).nonzero(as_tuple=False)         # [Nd,2]
        sy, sx = static_idx[:, 0], static_idx[:, 1]
        dy, dx = dyn_idx[:, 0], dyn_idx[:, 1]

        # pixel angular footprint → world-space radius at depth d
        fov_rad = math.radians(horizontal_fov)
        if projection == "EQUIRECTANGULAR":
            pix_angle = 2.0 * math.pi / W
        else:
            pix_angle = fov_rad / W
        opacity_logit = math.log(opacity / (1.0 - opacity))

        def make_splats(xyz: torch.Tensor, colors: torch.Tensor,
                        d: torch.Tensor, sfac: torch.Tensor) -> GaussianSplats:
            n = xyz.shape[0]
            scale = (d * pix_angle * sfac * splat_scale).clamp(min=1e-6).log()
            scale = scale.unsqueeze(-1).expand(n, 3).contiguous()
            rotation = torch.zeros(n, 4, device=xyz.device)
            rotation[:, 0] = 1.0
            op = torch.full((n, 1), opacity_logit, device=xyz.device)
            f_dc = (colors - 0.5) / C0
            f_rest = torch.zeros(n, 0, device=xyz.device)
            return GaussianSplats(xyz=xyz, scale=scale, rotation=rotation,
                                  opacity=op, f_dc=f_dc, f_rest=f_rest)

        sd = plate_d[sy, sx]
        static_splats = make_splats(dirs[sy, sx] * sd.unsqueeze(-1),
                                    plate_rgb[sy, sx], sd, stride_map[sy, sx])

        transparent_logit = math.log(0.01 / 0.99)
        splat_frames: List[GaussianSplats] = []
        dyn_dirs = dirs[dy, dx]
        dyn_sfac = stride_map[dy, dx]
        for t in range(T):
            dd = depth[t][dy, dx]
            dyn = make_splats(dyn_dirs * dd.unsqueeze(-1), rgb[t][dy, dx], dd,
                              dyn_sfac)
            if edge_threshold > 0 and dy.numel() > 0:
                # dynamic edges move per frame; keep the splat count constant
                # by turning edge splats transparent instead of dropping them
                e = edge_map(depth[t])[dy, dx]
                dyn.opacity[e] = transparent_logit
            splat_frames.append(GaussianSplats(
                xyz=torch.cat([static_splats.xyz, dyn.xyz]),
                scale=torch.cat([static_splats.scale, dyn.scale]),
                rotation=torch.cat([static_splats.rotation, dyn.rotation]),
                opacity=torch.cat([static_splats.opacity, dyn.opacity]),
                f_dc=torch.cat([static_splats.f_dc, dyn.f_dc]),
                f_rest=torch.cat([static_splats.f_rest, dyn.f_rest]),
            ))

        return (splat_frames, static_splats,
                int(static_splats.xyz.shape[0]), int(dyn_idx.shape[0]))


class RenderSplatSequence:
    """Render each frame's splat cloud along a camera trajectory
    (interpolated to the sequence length; identity if not provided)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splat_frames": ("GSPLAT_SEQ",),
                "camera_projection": (Projection.PROJECTIONS, {"default": "PINHOLE"}),
                "camera_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 360.0}),
                "output_width": ("INT", {"default": 768, "min": 8, "max": 8192}),
                "output_height": ("INT", {"default": 768, "min": 8, "max": 8192}),
                "render_mode": (["auto", "gsplat", "fast", "over"], {"default": "auto"}),
            },
            "optional": {
                "trajectory": ("TENSOR", {"tooltip": "[K,4,4]; interpolated to the frame count. Identity if absent."}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TENSOR")
    RETURN_NAMES = ("images", "masks", "disparities")
    FUNCTION = "render_sequence"
    CATEGORY = "Camera/Video4D"

    def render_sequence(
        self,
        splat_frames: List[GaussianSplats],
        camera_projection: str,
        camera_horizontal_fov: float,
        output_width: int,
        output_height: int,
        render_mode: str,
        trajectory: Optional[torch.Tensor] = None,
        device: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(splat_frames)
        if trajectory is None:
            poses = torch.eye(4).unsqueeze(0).expand(T, 4, 4)
        else:
            poses = interpolate_se3(trajectory, T)

        renderer = RenderSplat()
        images, masks, disps = [], [], []
        for t in range(T):
            img, mask, disp = renderer.render_splats(
                splats=splat_frames[t],
                camera_matrix=poses[t].cpu().numpy(),
                camera_projection=camera_projection,
                camera_horizontal_fov=camera_horizontal_fov,
                output_width=output_width,
                output_height=output_height,
                max_splats=0,
                opacity_is_logit=True,
                add_sh_bias=True,
                render_mode=render_mode,
                chunk_size=256,
                max_radius=32,
                device=device,
            )
            images.append(img.cpu())
            mask = mask if mask.dim() == 3 else mask.unsqueeze(0)
            masks.append(mask.cpu())
            disps.append(disp.cpu())
        return (torch.cat(images, dim=0), torch.cat(masks, dim=0),
                torch.cat(disps, dim=0))


class SaveSplats4DVideo:
    """
    Export the splat sequence as antimatter15 .splat frames and encode them
    into a single streamable .splat4d file with the `splat4d` CLI
    (pip install splats4d — https://github.com/adamraudonis/splats4D).
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splat_frames": ("GSPLAT_SEQ",),
                "filename_prefix": ("STRING", {"default": "splats4d/scene"}),
                "pos_mm": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0,
                                     "tooltip": "Position error bound (mm) — scene units are meters"}),
                "color_levels": ("INT", {"default": 4, "min": 1, "max": 64,
                                         "tooltip": "Color error bound (out of 255)"}),
                "scale_pct": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 50.0,
                                        "tooltip": "Scale error bound (%)"}),
                "gop": ("INT", {"default": 30, "min": 1, "max": 300}),
                "keep_frames": ("BOOLEAN", {"default": False,
                                            "tooltip": "Keep the raw per-frame .splat files next to the .splat4d"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Camera/Video4D"

    @staticmethod
    def _splat_bytes(s: GaussianSplats) -> bytes:
        """antimatter15 .splat rows: pos f32×3 | scale f32×3 | rgba u8×4 | rot u8×4."""
        n = s.xyz.shape[0]
        pos = s.xyz.detach().cpu().numpy().astype(np.float32)
        scale = torch.exp(s.scale).detach().cpu().numpy().astype(np.float32)
        rgb = (0.5 + C0 * s.f_dc.detach().cpu().numpy())
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        alpha = torch.sigmoid(s.opacity).detach().cpu().numpy()
        alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8).reshape(n, 1)
        quat = s.rotation.detach().cpu().numpy()
        quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
        rot = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)
        rows = np.zeros((n, 32), dtype=np.uint8)
        rows[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
        rows[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
        rows[:, 24:27] = rgb
        rows[:, 27:28] = alpha
        rows[:, 28:32] = rot
        return rows.tobytes()

    def save(
        self,
        splat_frames: List[GaussianSplats],
        filename_prefix: str,
        pos_mm: float,
        color_levels: int,
        scale_pct: float,
        gop: int,
        keep_frames: bool,
    ) -> Dict[str, Any]:
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, 0, 0)
        os.makedirs(full_output_folder, exist_ok=True)
        base = f"{filename}_{counter:05}"
        frames_dir = os.path.join(full_output_folder, base + "_frames")
        os.makedirs(frames_dir, exist_ok=True)

        for t, s in enumerate(splat_frames):
            with open(os.path.join(frames_dir, f"frame_{t:04d}.splat"), "wb") as f:
                f.write(self._splat_bytes(s))

        out_path = os.path.join(full_output_folder, base + ".splat4d")
        cli = shutil.which("splat4d")
        if cli is None:
            raise RuntimeError("splat4d CLI not found — pip install splats4d")
        cmd = [cli, "encode", "-i", frames_dir, "-o", out_path,
               "--pos-mm", str(pos_mm), "--color-levels", str(color_levels),
               "--scale-pct", str(scale_pct), "--gop", str(gop)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        if proc.returncode != 0:
            raise RuntimeError("splat4d encode failed:\n" + "\n".join(log_tail))
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

        n = len(splat_frames)
        per_frame = int(splat_frames[0].xyz.shape[0]) if n else 0
        size_mb = os.path.getsize(out_path) / 1e6
        info = (f"{out_path}\n{n} frames × {per_frame} splats → {size_mb:.1f} MB\n"
                + "\n".join(log_tail))
        print(f"[SaveSplats4DVideo] {info}")
        return {"ui": {"text": [info]}, "result": (info,)}


def _splats_slice(s: GaussianSplats, a: int, b: Optional[int] = None) -> GaussianSplats:
    b = s.xyz.shape[0] if b is None else b
    return GaussianSplats(xyz=s.xyz[a:b], scale=s.scale[a:b],
                          rotation=s.rotation[a:b], opacity=s.opacity[a:b],
                          f_dc=s.f_dc[a:b], f_rest=s.f_rest[a:b])


def _splats_concat(a: GaussianSplats, b: GaussianSplats) -> GaussianSplats:
    b = b.to(a.xyz.device)
    return GaussianSplats(
        xyz=torch.cat([a.xyz, b.xyz]), scale=torch.cat([a.scale, b.scale]),
        rotation=torch.cat([a.rotation, b.rotation]),
        opacity=torch.cat([a.opacity, b.opacity]),
        f_dc=torch.cat([a.f_dc, b.f_dc]),
        f_rest=torch.cat([a.f_rest, b.f_rest]))


class BakeRenderFill:
    """
    Lift generatively filled disocclusion pixels of a rendered novel view
    back into the STATIC splat layer, so the exported scene itself handles
    occlusions (instead of fixing them per rendered video).

    Pixels whose render coverage is below `hole_threshold` take their color
    from `filled_images` (e.g. the WAN-VACE-inpainted render) and their depth
    from the surrounding rendered background, continued by max-propagation
    (the disoccluded surface is behind whatever was in front). They are
    unprojected along the bake camera's rays and appended to the static base
    of every frame — the splat count stays constant across frames.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "splat_frames": ("GSPLAT_SEQ",),
                "static_count": ("INT", {"default": 0, "min": 0, "max": 2 ** 31,
                                         "forceInput": True,
                                         "tooltip": "static_count output of SplatFramesFromVideo (static splats come first in every frame)"}),
                "filled_images": ("IMAGE", {"tooltip": "Inpainted render (e.g. WAN VACE output), same camera as coverage/disparities"}),
                "coverage": ("MASK", {"tooltip": "masks output of RenderSplatSequence (accumulated alpha)"}),
                "disparities": ("TENSOR", {"tooltip": "disparities output of RenderSplatSequence"}),
                "camera_horizontal_fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0}),
                "bake_frame": ("INT", {"default": -1, "min": -4096, "max": 4096,
                                       "tooltip": "Which frame of the render to bake from (-1 = last = widest camera offset)"}),
                "hole_threshold": ("FLOAT", {"default": 0.85, "min": 0.05, "max": 1.0, "step": 0.01,
                                             "tooltip": "Pixels with render alpha below this are baked"}),
                "splat_scale": ("FLOAT", {"default": 1.3, "min": 0.05, "max": 10.0, "step": 0.05}),
                "opacity": ("FLOAT", {"default": 0.95, "min": 0.05, "max": 0.995, "step": 0.005}),
            },
            "optional": {
                "trajectory": ("TENSOR", {"tooltip": "Same trajectory the render used; identity if absent"}),
            },
        }

    RETURN_TYPES = ("GSPLAT_SEQ", "GSPLAT", "INT")
    RETURN_NAMES = ("splat_frames", "baked_splats", "baked_count")
    FUNCTION = "bake"
    CATEGORY = "Camera/Video4D"

    def bake(
        self,
        splat_frames: List[GaussianSplats],
        static_count: int,
        filled_images: torch.Tensor,
        coverage: torch.Tensor,
        disparities: torch.Tensor,
        camera_horizontal_fov: float,
        bake_frame: int,
        hole_threshold: float,
        splat_scale: float,
        opacity: float,
        trajectory: Optional[torch.Tensor] = None,
    ) -> Tuple[List[GaussianSplats], GaussianSplats, int]:
        T = len(splat_frames)
        dev = splat_frames[0].xyz.device
        idx = bake_frame % T
        img = filled_images[idx].to(dev).float()                  # [h,w,3]
        alpha = _to_thw(coverage)[idx].to(dev).float()            # [h,w]
        disp = _to_thw(disparities.squeeze(-1) if disparities.dim() == 4
                       else disparities)[idx].to(dev).float()     # [h,w]
        h, w = alpha.shape

        if trajectory is None:
            M = torch.eye(4, device=dev)
        else:
            M = interpolate_se3(trajectory, T)[idx].to(dev)       # world→camera
        Minv = torch.inverse(M)

        hole = alpha < hole_threshold
        if not bool(hole.any()):
            return (splat_frames, _splats_slice(splat_frames[0], 0, 0), 0)

        # ray depth of the rendered background: disparity = alpha / depth
        known = (alpha > 0.7) & (disp > 1e-6)
        ld = torch.where(known, (alpha / disp.clamp(min=1e-6)).clamp(min=1e-6).log(),
                         torch.full_like(disp, -20.0))
        km = known.float().unsqueeze(0).unsqueeze(0)
        df = ld.unsqueeze(0).unsqueeze(0)
        for _ in range(256):
            prop = F.max_pool2d(df, 9, stride=1, padding=4)
            newly = (F.max_pool2d(km, 9, stride=1, padding=4) > 1e-6) & (km < 0.5)
            df = torch.where(newly, prop, df)
            km = torch.where(newly, torch.ones_like(km), km)
            if bool((km > 0.5).all()):
                break
        depth = df[0, 0].exp()

        dirs = _unit_directions("PINHOLE", camera_horizontal_fov, h, w, dev)
        hy, hx = hole.nonzero(as_tuple=True)
        p_cam = dirs[hy, hx] * depth[hy, hx].unsqueeze(-1)
        p_world = p_cam @ Minv[:3, :3].T + Minv[:3, 3]

        pix_angle = math.radians(camera_horizontal_fov) / w
        n = p_world.shape[0]
        scale = (depth[hy, hx] * pix_angle * splat_scale).clamp(min=1e-6).log()
        rotation = torch.zeros(n, 4, device=dev)
        rotation[:, 0] = 1.0
        baked = GaussianSplats(
            xyz=p_world,
            scale=scale.unsqueeze(-1).expand(n, 3).contiguous(),
            rotation=rotation,
            opacity=torch.full((n, 1), math.log(opacity / (1.0 - opacity)), device=dev),
            f_dc=(img[hy, hx] - 0.5) / C0,
            f_rest=torch.zeros(n, 0, device=dev),
        )

        new_static = _splats_concat(_splats_slice(splat_frames[0], 0, static_count), baked)
        out_frames = [_splats_concat(new_static, _splats_slice(f, static_count))
                      for f in splat_frames]
        return (out_frames, baked, n)


class MedianFrames:
    """Temporal median of an image batch — collapses a (generatively
    inpainted) background video into one flicker-free plate."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {"required": {"frames": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "median"
    CATEGORY = "Camera/Video4D"

    def median(self, frames: torch.Tensor) -> Tuple[torch.Tensor]:
        return (frames.float().median(dim=0, keepdim=True).values,)


class BlendImagesByMask:
    """base×(1-mask) + fill×mask with optional feathering; `fill` is
    bilinearly resized to `base` if their sizes differ. Used to composite a
    WAN/Flux-inpainted background into the static plate only where the true
    background was never observed."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base": ("IMAGE",),
                "fill": ("IMAGE",),
                "mask": ("MASK",),
                "feather": ("INT", {"default": 8, "min": 0, "max": 128,
                                    "tooltip": "Blur radius on the mask edge for a seamless blend"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"
    CATEGORY = "Camera/Video4D"

    def blend(self, base: torch.Tensor, fill: torch.Tensor,
              mask: torch.Tensor, feather: int) -> Tuple[torch.Tensor]:
        base = base.float()
        H, W = base.shape[1], base.shape[2]
        fill = fill.float()[:1]
        if fill.shape[1] != H or fill.shape[2] != W:
            fill = F.interpolate(fill.permute(0, 3, 1, 2), size=(H, W),
                                 mode="bilinear", align_corners=False
                                 ).permute(0, 2, 3, 1)
        m = _to_thw(mask).float()[:1].to(base.device)             # [1,H,W]
        if m.shape[1] != H or m.shape[2] != W:
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear",
                              align_corners=False).squeeze(1)
        if feather > 0:
            k = 2 * feather + 1
            m = F.avg_pool2d(m.unsqueeze(1), k, stride=1, padding=feather).squeeze(1)
        m = m.unsqueeze(-1).clamp(0, 1)
        return (base * (1 - m) + fill.to(base.device) * m,)


class ScaleMasks:
    """Resize a [T,H,W] mask batch (bilinear + threshold keeps edges clean)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "masks": ("MASK",),
                "width": ("INT", {"default": 1024, "min": 8, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 8192}),
                "binarize": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Binarization threshold. E.g. 0.12 on an inverted render-coverage mask also catches semi-transparent 'veil' pixels, not just holes."}),
                "repeat_to": ("INT", {"default": 0, "min": 0, "max": 4096,
                                      "tooltip": "Repeat a single mask to this many frames (0 = keep count). WanVaceToVideo pads missing mask frames with 1.0 (full regen), so repeat explicitly."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "scale"
    CATEGORY = "Camera/Video4D"

    def scale(self, masks: torch.Tensor, width: int, height: int,
              binarize: bool, threshold: float = 0.25,
              repeat_to: int = 0) -> Tuple[torch.Tensor]:
        m = _to_thw(masks).float()
        m = F.interpolate(m.unsqueeze(1), size=(height, width),
                          mode="bilinear", align_corners=False).squeeze(1)
        if binarize:
            m = (m > threshold).float()
        if repeat_to > m.shape[0]:
            m = m.expand(repeat_to, -1, -1) if m.shape[0] == 1 else torch.cat(
                [m, m[-1:].expand(repeat_to - m.shape[0], -1, -1)])
        return (m.contiguous(),)


NODE_CLASS_MAPPINGS = {
    "TemporalStaticPlate": TemporalStaticPlate,
    "SplatFramesFromVideo": SplatFramesFromVideo,
    "RenderSplatSequence": RenderSplatSequence,
    "BakeRenderFill": BakeRenderFill,
    "SaveSplats4DVideo": SaveSplats4DVideo,
    "MedianFrames": MedianFrames,
    "BlendImagesByMask": BlendImagesByMask,
    "ScaleMasks": ScaleMasks,
}
