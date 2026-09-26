#
# fur-gs: fur-region guided densification (D13, D14). Only active with --fur_densify.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# Per-Gaussian statistics (GaussianModel.fur_stats, N x STATS_DIM), accumulated over training views
# from the fur maps of tools/fur_detect (RGB-only detector):
#   0 fringe_sum, 1 interior_sum, 2 count           -> mean fur score at the projected center
#   3..8 M (xx, xy, xz, yy, yz, zz)                   -> sum of w * n n^T, n = normal of the plane that
#                                                        contains the camera ray and the 2D strand direction
#   9 weight_sum, 10..12 ray_sum                       -> mean viewing direction (degeneracy check)
# The 3D strand direction is the unit vector most orthogonal to all plane normals (smallest eigenvector of M).
#
import math
import os

import numpy as np
import torch
from PIL import Image

STATS_DIM = 13
MAP_NAMES = ("fringe", "interior", "theta", "coh")


def _load_map(path, size, nearest):
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != size:
            im = im.resize(size, Image.NEAREST if nearest else Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())


def quat_from_matrix(R):
    """(B,3,3) rotation matrices -> (B,4) quaternions (w, x, y, z), the convention of build_rotation."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    w = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) / 2
    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)
    q = torch.stack([w, x, y, z], -1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


class FurGuide:
    def __init__(self, opt, dataset, train_cameras):
        self.mode = opt.fur_mode
        if self.mode not in ("both", "fringe", "interior"):
            raise ValueError(f"--fur_mode must be both|fringe|interior, got {self.mode}")
        self.beta = opt.fur_beta
        self.decay = opt.fur_decay
        self.orient = opt.fur_orient_split
        self.aspect = opt.fur_orient_aspect
        self.min_conf = opt.fur_orient_min_conf
        self.min_interior = opt.fur_orient_min_interior
        map_dir = opt.fur_maps or os.path.join(dataset.source_path, "fur_maps")
        if not os.path.isdir(map_dir):
            raise FileNotFoundError(f"[fur] fur map folder not found: {map_dir} (run tools/fur_detect/detect.py)")
        self.maps = {}
        for cam in train_cameras:
            stem = os.path.splitext(cam.image_name)[0]
            size = (cam.image_width, cam.image_height)
            chans = [_load_map(os.path.join(map_dir, f"{stem}_{k}.png"), size, nearest=(k == "theta"))
                     for k in MAP_NAMES]
            self.maps[cam.image_name] = torch.stack(chans).cuda()      # (4,H,W) uint8
        self.n_oriented = 0
        self.n_split_candidates = 0
        print(f"[fur] maps from {map_dir} for {len(self.maps)} train views; mode={self.mode} beta={self.beta} "
              f"decay={self.decay} orient_split={self.orient}")

    def attach(self, gaussians):
        gaussians.fur_stats = torch.zeros((gaussians.get_xyz.shape[0], STATS_DIM), device="cuda")
        gaussians.fur_guide = self if self.orient else None

    @torch.no_grad()
    def accumulate(self, gaussians, cam, visibility_filter):
        maps = self.maps.get(cam.image_name)
        if maps is None:
            return
        idx = torch.nonzero(visibility_filter, as_tuple=True)[0]
        if idx.numel() == 0:
            return
        xyz = gaussians.get_xyz[idx]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1)
        p = xyz_h @ cam.full_proj_transform
        w = p[:, 3:4]
        ndc = p[:, :2] / (w + 1e-7)
        H, W = maps.shape[1], maps.shape[2]
        px = ((ndc[:, 0] + 1) * W - 1) * 0.5
        py = ((ndc[:, 1] + 1) * H - 1) * 0.5
        col, row = px.round().long(), py.round().long()
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (w[:, 0] > 0)
        idx, col, row, xyz_h = idx[ok], col[ok], row[ok], xyz_h[ok]
        v = maps[:, row, col].float() / 255.0          # (4, n)
        st = gaussians.fur_stats
        st[idx, 0] += v[0]
        st[idx, 1] += v[1]
        st[idx, 2] += 1
        if not self.orient:
            return
        theta = v[2] * math.pi
        wgt = v[1] * v[3]                              # interior fur score x coherence
        p_cam = (xyz_h @ cam.world_view_transform)[:, :3]
        ray = p_cam / p_cam.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], 1)
        n = torch.cross(ray, d, dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rot = cam.world_view_transform[:3, :3]         # row-vector convention: p_cam = p_w @ rot
        n_w = n @ rot.T
        ray_w = ray @ rot.T
        a = wgt[:, None]
        st[idx, 3] += a[:, 0] * n_w[:, 0] * n_w[:, 0]
        st[idx, 4] += a[:, 0] * n_w[:, 0] * n_w[:, 1]
        st[idx, 5] += a[:, 0] * n_w[:, 0] * n_w[:, 2]
        st[idx, 6] += a[:, 0] * n_w[:, 1] * n_w[:, 1]
        st[idx, 7] += a[:, 0] * n_w[:, 1] * n_w[:, 2]
        st[idx, 8] += a[:, 0] * n_w[:, 2] * n_w[:, 2]
        st[idx, 9] += wgt
        st[idx, 10:13] += a * ray_w

    def scores(self, gaussians):
        st = gaussians.fur_stats
        cnt = st[:, 2].clamp(min=1e-6)
        fr, it = st[:, 0] / cnt, st[:, 1] / cnt
        fr = torch.where(st[:, 2] > 0, fr, torch.zeros_like(fr))
        it = torch.where(st[:, 2] > 0, it, torch.zeros_like(it))
        return fr, it

    @torch.no_grad()
    def grad_scale(self, gaussians):
        """Per-Gaussian multiplier of the view-space gradient: 1 + beta * fur score (= lower densify threshold)."""
        fr, it = self.scores(gaussians)
        s = {"fringe": fr, "interior": it, "both": torch.maximum(fr, it)}[self.mode]
        return (1.0 + self.beta * s)[:, None]

    @torch.no_grad()
    def after_densify(self, gaussians):
        """Exponential forgetting so that statistics follow Gaussians that move during training."""
        gaussians.fur_stats *= self.decay

    @torch.no_grad()
    def strand_dirs(self, stats):
        """Smallest eigenvector of M -> 3D strand direction, confidence 1 - l0/l1, and |t . mean ray|."""
        m = stats[:, 3:9]
        M = torch.stack([m[:, 0], m[:, 1], m[:, 2], m[:, 1], m[:, 3], m[:, 4], m[:, 2], m[:, 4], m[:, 5]], 1).view(-1, 3, 3)
        wsum = stats[:, 9].clamp(min=1e-8)
        M = M / wsum[:, None, None]
        evals, evecs = torch.linalg.eigh(M)
        t = evecs[:, :, 0]
        conf = 1 - evals[:, 0] / evals[:, 1].clamp(min=1e-8)
        ray = stats[:, 10:13] / wsum[:, None]
        ray = ray / ray.norm(dim=1, keepdim=True).clamp(min=1e-8)
        along_ray = (t * ray).sum(1).abs()
        return t, conf, along_ray

    @torch.no_grad()
    def oriented_split(self, gaussians, selected, N, new_xyz, new_scaling, new_rotation):
        """Replace the default children of confident fur-interior Gaussians by children stretched along the strand.
        new_* are the default children (N stacked copies of the selected set, activated scaling)."""
        stats = gaussians.fur_stats[selected]
        n_sel = stats.shape[0]
        self.n_split_candidates += n_sel
        if n_sel == 0:
            return new_xyz, new_scaling, new_rotation
        cnt = stats[:, 2].clamp(min=1e-6)
        interior = stats[:, 1] / cnt
        t, conf, along_ray = self.strand_dirs(stats)
        use = (interior >= self.min_interior) & (conf >= self.min_conf) & (along_ray < 0.7) & (stats[:, 9] > 0)
        if not bool(use.any()):
            return new_xyz, new_scaling, new_rotation
        self.n_oriented += int(use.sum())
        s = gaussians.get_scaling[selected][use]                    # parent scales (k,3)
        Rp = gaussians.get_rotation[selected][use]
        from utils.general_utils import build_rotation
        Rp = build_rotation(Rp)                                     # (k,3,3), columns = parent axes
        tu = t[use]
        # second axis: the parent axis least aligned with t, orthogonalised
        dots = (Rp * tu[:, :, None]).sum(1).abs()                   # (k,3) |axis_j . t|
        j = dots.argmin(1)
        a = Rp[torch.arange(len(j)), :, j]
        u = a - (a * tu).sum(1, keepdim=True) * tu
        u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-8)
        wv = torch.cross(tu, u, dim=1)
        Rc = torch.stack([tu, u, wv], 2)                            # columns t, u, w
        q = quat_from_matrix(Rc)
        s_sorted = torch.sort(s, dim=1, descending=True).values
        s_long = s_sorted[:, 0] / (0.8 * N)
        s_across = torch.minimum(s_sorted[:, 1] / (0.8 * N), s_long * self.aspect)
        s_thin = torch.minimum(s_sorted[:, 2] / (0.8 * N), s_across)
        child_scale = torch.stack([s_long, s_across, s_thin], 1)
        k = s.shape[0]
        use_rep = use.repeat(N)
        eps = torch.randn((N * k, 1), device="cuda")
        centers = gaussians.get_xyz[selected][use].repeat(N, 1)
        new_xyz = new_xyz.clone()
        new_scaling = new_scaling.clone()
        new_rotation = new_rotation.clone()
        new_xyz[use_rep] = centers + tu.repeat(N, 1) * eps * s_sorted[:, 0].repeat(N)[:, None]
        new_scaling[use_rep] = child_scale.repeat(N, 1)
        new_rotation[use_rep] = q.repeat(N, 1)
        return new_xyz, new_scaling, new_rotation
