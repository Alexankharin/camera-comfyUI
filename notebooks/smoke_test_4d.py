"""Standalone CPU smoke test for the 4D-world node stack (no ComfyUI, no CUDA,
no model downloads).

Run with:
    python notebooks/smoke_test_4d.py

Stubs `folder_paths` via sys.modules injection so the repo modules import
outside the ComfyUI runtime, then functionally exercises the NEW code paths
with small synthetic data:

 1. interpolate_se3        (pointcloud_nodes, contract C1)
 2. render_gaussians       (GS_nodes, contract C2) shapes + empty case
 3. render_gaussians fast  anisotropic footprint
 4. GaussianSplats4D.at_time (GS4D_nodes, contract C3)
 5. BuildSplats4D          kNN track binding
 6. SplitSplatsByMask
 7. MotionMaskFromDepth
 8. align_depth_scale (world_nodes, contract C4) + DepthEdgeFilter
 9. FuseSplats             weighted voxel fusion
10. SphereSplatSeed        pano -> splat sphere -> render round-trip
"""

import math
import os
import sys
import tempfile
import traceback
import types

# --------------------------------------------------------------------------- #
# Environment setup: repo on sys.path + folder_paths stub (before repo imports)
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_TMP_DIR = tempfile.mkdtemp(prefix="smoke_test_4d_")


def _stub_get_save_image_path(filename_prefix, output_dir, *args, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, filename_prefix, 0, "", filename_prefix


_fp_stub = types.ModuleType("folder_paths")
_fp_stub.get_input_directory = lambda: _TMP_DIR
_fp_stub.get_output_directory = lambda: _TMP_DIR
_fp_stub.get_temp_directory = lambda: _TMP_DIR
_fp_stub.get_save_image_path = _stub_get_save_image_path
_fp_stub.get_annotated_filepath = lambda name: os.path.join(_TMP_DIR, name)
_fp_stub.exists_annotated_filepath = lambda name: os.path.exists(os.path.join(_TMP_DIR, name))
_fp_stub.get_filename_list = lambda folder: []
_fp_stub.models_dir = _TMP_DIR
sys.modules["folder_paths"] = _fp_stub

import numpy as np  # noqa: E402
import torch  # noqa: E402

import GS_nodes  # noqa: E402
import GS4D_nodes  # noqa: E402
import pointcloud_nodes  # noqa: E402
import world_nodes  # noqa: E402

GaussianSplats = GS_nodes.GaussianSplats

torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_splats(
    xyz: torch.Tensor,
    sigma: float = 0.05,
    color: tuple = None,
    opacity_logit: float = 4.0,
) -> GaussianSplats:
    """Isotropic sh_order-0 splats at the given positions."""
    n = xyz.shape[0]
    if color is None:
        rgb = torch.rand(n, 3)
    else:
        rgb = torch.tensor(color, dtype=torch.float32).view(1, 3).expand(n, 3)
    C0 = 0.28209479177387814
    return GaussianSplats(
        xyz=xyz.float(),
        scale=torch.full((n, 3), math.log(sigma)),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 4).expand(n, 4).contiguous(),
        opacity=torch.full((n, 1), float(opacity_logit)),
        f_dc=((rgb - 0.5) / C0).contiguous(),
        f_rest=torch.zeros(n, 0),
        sh_order=0,
    )


def rot_x(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    return torch.tensor(
        [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]],
        dtype=torch.float32,
    )


def rot_y(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    return torch.tensor(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]],
        dtype=torch.float32,
    )


def make_pose(R: torch.Tensor, t) -> torch.Tensor:
    M = torch.eye(4)
    M[:3, :3] = R
    M[:3, 3] = torch.tensor(t, dtype=torch.float32)
    return M


IDENTITY_4X4 = torch.eye(4)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_01_interpolate_se3():
    poses = torch.stack(
        [
            make_pose(torch.eye(3), [0.0, 0.0, 0.0]),
            make_pose(rot_y(90.0), [1.0, 2.0, 3.0]),
            make_pose(rot_y(90.0) @ rot_x(45.0), [-1.0, 0.0, 2.0]),
        ]
    )
    out = pointcloud_nodes.interpolate_se3(poses, 10)
    assert out.shape == (10, 4, 4), f"shape {tuple(out.shape)}"

    eye = torch.eye(3)
    for i in range(10):
        R = out[i, :3, :3]
        ortho_err = (R @ R.T - eye).abs().max().item()
        det = torch.det(R).item()
        assert ortho_err < 1e-4, f"step {i}: R@R.T deviates from I by {ortho_err}"
        assert abs(det - 1.0) < 1e-4, f"step {i}: det(R)={det}"
        assert torch.allclose(out[i, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)

    assert (out[0] - poses[0]).abs().max().item() < 1e-4, "start pose mismatch"
    assert (out[-1] - poses[-1]).abs().max().item() < 1e-4, "end pose mismatch"

    # K == 1 repeats.
    rep = pointcloud_nodes.interpolate_se3(poses[:1], 5)
    assert rep.shape == (5, 4, 4)
    assert (rep - poses[0]).abs().max().item() < 1e-6


def test_02_render_gaussians_shapes_and_empty():
    n, H, W = 200, 48, 64
    xyz = torch.stack(
        [
            torch.rand(n) * 2.0 - 1.0,
            torch.rand(n) * 2.0 - 1.0,
            torch.rand(n) * 3.0 + 2.0,
        ],
        dim=-1,
    )
    splats = make_splats(xyz, sigma=0.05)

    for projection, fov in (("PINHOLE", 90.0), ("EQUIRECTANGULAR", 360.0)):
        image, mask, disparity = GS_nodes.render_gaussians(
            splats, IDENTITY_4X4, projection, fov, W, H,
            render_mode="fast", device="cpu",
        )
        assert image.shape == (1, H, W, 3), f"{projection} image {tuple(image.shape)}"
        assert mask.shape == (H, W), f"{projection} mask {tuple(mask.shape)}"
        assert disparity.shape == (1, H, W, 1), f"{projection} disparity {tuple(disparity.shape)}"
        assert torch.isfinite(image).all() and torch.isfinite(disparity).all()
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0 + 1e-6
        assert float(mask.sum()) > 0.0, f"{projection}: nothing rendered"

    # Empty case: every splat strictly behind a pinhole camera (known past bug:
    # early return used to yield only 2 outputs).
    behind = make_splats(xyz * torch.tensor([1.0, 1.0, -1.0]), sigma=0.05)
    result = GS_nodes.render_gaussians(
        behind, IDENTITY_4X4, "PINHOLE", 90.0, W, H,
        render_mode="fast", device="cpu",
    )
    assert isinstance(result, tuple) and len(result) == 3, f"empty render returned {len(result)} outputs"
    image, mask, disparity = result
    assert image.shape == (1, H, W, 3)
    assert mask.shape == (H, W)
    assert disparity.shape == (1, H, W, 1)
    assert float(mask.sum()) == 0.0


def test_03_fast_mode_anisotropy():
    H = W = 128
    ang = math.radians(45.0) / 2.0
    splats = GaussianSplats(
        xyz=torch.tensor([[0.0, 0.0, 3.0]]),
        scale=torch.log(torch.tensor([[0.5, 0.01, 0.01]])),
        rotation=torch.tensor([[math.cos(ang), 0.0, 0.0, math.sin(ang)]]),  # 45 deg about +z
        opacity=torch.tensor([[6.0]]),
        f_dc=torch.zeros(1, 3),
        f_rest=torch.zeros(1, 0),
        sh_order=0,
    )
    image, mask, disparity = GS_nodes.render_gaussians(
        splats, IDENTITY_4X4, "PINHOLE", 60.0, W, H,
        render_mode="fast", max_radius=64, device="cpu",
    )
    assert float(mask.sum()) > 0.0, "elongated splat rendered nothing"

    # Alpha-weighted pixel covariance of the footprint.
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    w = mask.flatten()
    wsum = w.sum()
    mx = (w * xs.flatten()).sum() / wsum
    my = (w * ys.flatten()).sum() / wsum
    dx = xs.flatten() - mx
    dy = ys.flatten() - my
    cxx = (w * dx * dx).sum() / wsum
    cyy = (w * dy * dy).sum() / wsum
    cxy = (w * dx * dy).sum() / wsum
    cov = torch.tensor([[cxx, cxy], [cxy, cyy]])
    evals, evecs = torch.linalg.eigh(cov)
    ratio = float(evals[1] / evals[0].clamp(min=1e-8))
    assert ratio > 2.0, f"footprint not elongated: eigenvalue ratio {ratio:.2f}"

    # Principal axis should be near 45 degrees (rotation honored).
    major = evecs[:, 1]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    assert abs(angle - 45.0) < 15.0, f"major axis at {angle:.1f} deg, expected ~45"


def test_04_at_time():
    T = 5
    canonical = make_splats(torch.tensor([[0.0, 0.0, 2.0], [0.0, 1.0, 3.0]]))
    static = make_splats(torch.tensor([[5.0, 5.0, 5.0]]))
    start = torch.tensor([[0.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    end = torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 3.0]])
    ts = torch.linspace(0.0, 1.0, T)
    trajectories = torch.stack([start + (end - start) * t for t in ts])  # [5,2,3]

    s4d = GS4D_nodes.GaussianSplats4D(
        static=static, canonical=canonical, trajectories=trajectories, times=ts,
    )

    mid = s4d.at_time(0.5)
    assert len(mid) == 3, f"count {len(mid)} != dynamic+static (3)"
    # Concat order is [static, dynamic].
    assert torch.allclose(mid.xyz[0], static.xyz[0], atol=1e-6)
    expected_mid = 0.5 * (start + end)
    assert torch.allclose(mid.xyz[1:], expected_mid, atol=1e-5), (
        f"midpoint mismatch: {mid.xyz[1:]} vs {expected_mid}"
    )

    lo = s4d.at_time(-1.0)
    hi = s4d.at_time(2.0)
    assert torch.allclose(lo.xyz[1:], start, atol=1e-5), "t<range should clamp to first step"
    assert torch.allclose(hi.xyz[1:], end, atol=1e-5), "t>range should clamp to last step"


def test_05_build_splats4d():
    T = 5
    ts = torch.linspace(0.0, 1.0, T)
    # Two control tracks moving apart along x.
    track_a = torch.stack([torch.tensor([-1.0 - 2.0 * t, 0.0, 2.0]) for t in ts])
    track_b = torch.stack([torch.tensor([1.0 + 2.0 * t, 0.0, 2.0]) for t in ts])
    trajectories3d = torch.stack([track_a, track_b], dim=1)  # [T,2,3]

    canonical = make_splats(torch.tensor([[-1.05, 0.0, 2.0], [1.05, 0.0, 2.0]]))
    node = GS4D_nodes.BuildSplats4D()
    (s4d,) = node.build_splats4d(
        canonical=canonical,
        trajectories3d=trajectories3d,
        reference_index=0,
        knn=1,
        rbf_gamma=0.0,
        device="cpu",
    )
    traj = s4d.trajectories
    assert traj.shape == (T, 2, 3), f"trajectories shape {tuple(traj.shape)}"
    # Reference timestep: splats stay at their canonical positions.
    assert torch.allclose(traj[0], canonical.xyz, atol=1e-5)
    # Each splat follows its nearest track's displacement direction.
    disp0 = traj[-1, 0] - traj[0, 0]
    disp1 = traj[-1, 1] - traj[0, 1]
    assert disp0[0] < -1.0, f"splat 0 should move -x with track A, moved {disp0.tolist()}"
    assert disp1[0] > 1.0, f"splat 1 should move +x with track B, moved {disp1.tolist()}"
    assert torch.allclose(traj[-1, 0], torch.tensor([-3.05, 0.0, 2.0]), atol=1e-4)
    assert torch.allclose(traj[-1, 1], torch.tensor([3.05, 0.0, 2.0]), atol=1e-4)


def test_06_split_splats_by_mask():
    H = W = 32
    mask = torch.zeros(H, W)
    mask[:, : W // 2] = 1.0  # left half white

    # 10 splats projecting into the left half (x<0), 10 into the right half,
    # 5 behind the camera.
    jitter = torch.linspace(-0.1, 0.1, 10)
    left = torch.stack([torch.full((10,), -0.5) + jitter * 0.1, jitter, torch.full((10,), 2.0)], dim=-1)
    right = torch.stack([torch.full((10,), 0.5) + jitter * 0.1, jitter, torch.full((10,), 2.0)], dim=-1)
    behind = torch.stack([jitter[:5], jitter[:5], torch.full((5,), -2.0)], dim=-1)
    splats = make_splats(torch.cat([left, right, behind], dim=0))

    node = GS4D_nodes.SplitSplatsByMask()
    inside, outside = node.split_splats(
        splats=splats,
        mask=mask,
        projection="PINHOLE",
        horizontal_fov=90.0,
        threshold=0.5,
        camera_matrix=None,
        device="cpu",
    )
    assert len(inside) == 10, f"inside count {len(inside)} != 10"
    assert len(outside) == 15, f"outside count {len(outside)} != 15 (10 right + 5 behind)"
    assert (inside.xyz[:, 0] < 0).all(), "inside splats should be the x<0 group"


def test_07_motion_mask_from_depth():
    T, H, W = 6, 32, 32
    depth = torch.full((T, H, W), 5.0)
    r0, r1 = 8, 16
    for t in range(T):
        depth[t, r0:r1, r0:r1] = 3.0 + 0.4 * t  # depth-changing square patch

    poses = torch.eye(4).unsqueeze(0).expand(T, 4, 4).contiguous()
    node = GS4D_nodes.MotionMaskFromDepth()
    (mask,) = node.motion_mask(
        depth_seq=depth,
        trajectory=poses,
        input_projection="PINHOLE",
        input_horizontal_fov=90.0,
        threshold=0.10,
        frame_gap=2,
        dilate=0,
        device="cpu",
    )
    assert mask.shape == (T, H, W), f"mask shape {tuple(mask.shape)}"

    patch = mask[:, r0:r1, r0:r1]
    background = mask.clone()
    background[:, r0:r1, r0:r1] = 0.0
    patch_mean = float(patch.mean())
    bg_sum = float(background.sum())
    assert patch_mean > 0.9, f"moving square under-detected: mean {patch_mean:.3f}"
    assert bg_sum == 0.0, f"static plane falsely flagged: {bg_sum} pixels"


def test_08_align_depth_scale_and_depth_edge_filter():
    H = W = 32
    new_depth = torch.rand(H, W) * 9.0 + 1.0
    # ref disparity = 0.5 * new disparity + 0.1  (i.e. ref = 2*new before shift).
    true_scale, true_shift = 0.5, 0.1
    ref_depth = 1.0 / (true_scale / new_depth + true_shift)
    valid = torch.ones(H, W)

    aligned, scale, shift = world_nodes.align_depth_scale(
        new_depth, ref_depth, valid, mode="scale_shift"
    )
    assert abs(scale - true_scale) / true_scale < 0.05, f"scale {scale} vs {true_scale}"
    assert abs(shift - true_shift) / true_shift < 0.05, f"shift {shift} vs {true_shift}"
    rel_err = float(((aligned - ref_depth).abs() / ref_depth).max())
    assert rel_err < 0.01, f"aligned depth off by {rel_err:.4f} (rel)"

    # DepthEdgeFilter: a vertical step edge must be masked out, flat kept.
    depth = torch.full((H, W), 1.0)
    depth[:, W // 2 :] = 5.0
    node = pointcloud_nodes.DepthEdgeFilter()
    (valid_mask,) = node.filter_edges(depth, relative_threshold=0.05, dilate=1)
    assert valid_mask.shape == (H, W)
    edge_cols = valid_mask[:, W // 2 - 1 : W // 2 + 1]
    assert float(edge_cols.max()) == 0.0, "step-edge pixels not masked out"
    assert float(valid_mask[:, : W // 2 - 3].min()) == 1.0, "flat left region wrongly masked"
    assert float(valid_mask[:, W // 2 + 3 :].min()) == 1.0, "flat right region wrongly masked"


def test_09_fuse_splats():
    n = 20
    voxel = 0.5
    base = torch.stack(
        [
            torch.arange(n, dtype=torch.float32) * voxel + 0.15,
            torch.full((n,), 0.15),
            torch.full((n,), 0.15),
        ],
        dim=-1,
    )
    cloud_a = make_splats(base)
    cloud_b = make_splats(base + 0.2)  # same voxels as A (0.15+0.2 < 0.5)

    node = GS_nodes.FuseSplats()
    (fused,) = node.fuse_splats(cloud_a, cloud_b, voxel, "smart", 1.0, 1.0, device="cpu")
    assert len(fused) < len(cloud_a) + len(cloud_b), (
        f"voxel fuse did not reduce: {len(fused)} vs {len(cloud_a) + len(cloud_b)}"
    )
    assert len(fused) == n, f"expected one splat per voxel ({n}), got {len(fused)}"

    # Strong weight_a pulls fused positions onto cloud A.
    (fused_w,) = node.fuse_splats(cloud_a, cloud_b, voxel, "average", 1000.0, 1.0, device="cpu")
    assert len(fused_w) == n
    d_a = torch.cdist(fused_w.xyz, cloud_a.xyz).min(dim=1).values
    d_b = torch.cdist(fused_w.xyz, cloud_b.xyz).min(dim=1).values
    assert float(d_a.max()) < 0.01, f"fused positions not near cloud A (max dist {float(d_a.max()):.4f})"
    assert (d_a < d_b).all(), "weight_a=1000 should pull fused splats toward cloud A"


def test_10_sphere_splat_seed():
    H, W = 64, 128
    stride = 2
    color = (0.2, 0.6, 0.9)
    pano = torch.tensor(color).view(1, 1, 1, 3).expand(1, H, W, 3).contiguous()

    node = world_nodes.SphereSplatSeed()
    (splats,) = node.seed_sphere(
        image=pano,
        horizontal_fov=360.0,
        radius=5.0,
        splat_scale_frac=1.5,
        stride=stride,
        device="cpu",
    )
    expected = (H // stride) * (W // stride)
    assert abs(len(splats) - expected) <= max(4, expected // 20), (
        f"splat count {len(splats)} far from expected ~{expected}"
    )

    image, mask, disparity = GS_nodes.render_gaussians(
        splats, IDENTITY_4X4, "PINHOLE", 60.0, 64, 64,
        render_mode="fast", device="cpu",
    )
    assert float(mask.sum()) > 0.0, "pinhole render of the sphere seed is empty"
    solid = mask > 0.9
    assert bool(solid.any()), "no confidently covered pixels in the render"
    rendered = image[0][solid]  # [K,3]
    target = torch.tensor(color)
    err = (rendered.mean(dim=0) - target).abs().max().item()
    assert err < 0.05, f"color round-trip failed: rendered mean {rendered.mean(dim=0).tolist()} vs {color}"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
TESTS = [
    test_01_interpolate_se3,
    test_02_render_gaussians_shapes_and_empty,
    test_03_fast_mode_anisotropy,
    test_04_at_time,
    test_05_build_splats4d,
    test_06_split_splats_by_mask,
    test_07_motion_mask_from_depth,
    test_08_align_depth_scale_and_depth_edge_filter,
    test_09_fuse_splats,
    test_10_sphere_splat_seed,
]


def main() -> int:
    passed = 0
    failed = []
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[ ok ] {name}")
    print(f"\n{passed}/{len(TESTS)} tests passed")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
