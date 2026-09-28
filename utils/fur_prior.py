#
# fur-gs: fur-region Gaussian shape prior (docs/gauss_prior.md, method_ideas 6). Only active with --fur_len_prior.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# Prior: the pixel length of the longest axis of fur Gaussians in a DENSE reconstruction of ANOTHER animal/scene
# (tools/eval/gauss_shape.py stats -> gauss_shape.json, quantile --fur_len_q). Using the dense model of the same
# scene would leak the test cameras.
# During training every axis of a fur Gaussian is clamped to cap_px * z / fx after each optimizer step, where z is the
# depth in the nearest train camera (center distance). Fur = mean fringe/interior score of the projected center over
# all train views > FUR_TH (same projection as utils/fur_utils.py, occlusion ignored).
#
import json
import math
import os

import torch

from utils.fur_utils import _load_map

FUR_TH = 0.3


class FurLenPrior:
    def __init__(self, opt, dataset, train_cameras):
        with open(opt.fur_len_prior, encoding="utf-8") as f:
            pj = json.load(f)
        self.cap_px = float(pj["groups"]["fur"]["len_px"][f"p{opt.fur_len_q}"])
        self.from_iter = opt.fur_len_from
        self.every = opt.fur_len_every
        map_dir = opt.fur_maps or os.path.join(dataset.source_path, "fur_maps")
        maps, P, C, fx, V = [], [], [], [], []
        for cam in train_cameras:
            stem = os.path.splitext(cam.image_name)[0]
            size = (cam.image_width, cam.image_height)
            maps.append(torch.stack([_load_map(os.path.join(map_dir, f"{stem}_{k}.png"), size, nearest=False)
                                     for k in ("fringe", "interior")]))
            P.append(cam.full_proj_transform)
            V.append(cam.world_view_transform)
            C.append(cam.camera_center)
            fx.append(cam.image_width / (2 * math.tan(cam.FoVx / 2)))
        self.maps = torch.stack(maps).cuda()                      # (K,2,H,W) uint8
        self.P, self.V = torch.stack(P).cuda(), torch.stack(V).cuda()
        self.C = torch.stack(C).cuda()
        self.fx = torch.tensor(fx, device="cuda")
        self.mask, self.log_cap, self.n = None, None, -1
        self.n_clamped = 0
        print(f"[fur_len] prior {opt.fur_len_prior} p{opt.fur_len_q}: cap {self.cap_px:.2f} px, from iter {self.from_iter}, "
              f"{len(train_cameras)} train views")

    @torch.no_grad()
    def update(self, gaussians):
        xyz = gaussians.get_xyz
        n = xyz.shape[0]
        H, W = self.maps.shape[2], self.maps.shape[3]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1)
        score_sum = torch.zeros(n, 2, device="cuda")
        cnt = torch.zeros(n, device="cuda")
        for k in range(self.P.shape[0]):
            p = xyz_h @ self.P[k]
            w = p[:, 3]
            px = ((p[:, 0] / (w + 1e-7) + 1) * W - 1) * 0.5
            py = ((p[:, 1] / (w + 1e-7) + 1) * H - 1) * 0.5
            col, row = px.round().long(), py.round().long()
            ok = (col >= 0) & (col < W) & (row >= 0) & (row < H) & (w > 0)
            idx = torch.nonzero(ok, as_tuple=True)[0]
            score_sum[idx] += self.maps[k][:, row[idx], col[idx]].float().T / 255.0
            cnt[idx] += 1
        score = score_sum / cnt.clamp(min=1)[:, None]
        self.mask = score.max(1).values > FUR_TH
        nearest = torch.cdist(xyz, self.C).argmin(1)                # nearest train camera (center distance)
        z = (xyz_h[:, None, :] @ self.V[nearest]).squeeze(1)[:, 2]  # depth in that camera
        self.log_cap = torch.log(self.cap_px * z.clamp(min=1e-6) / self.fx[nearest])
        self.n = n

    @torch.no_grad()
    def apply(self, gaussians, iteration):
        if iteration < self.from_iter:
            return
        n = gaussians.get_xyz.shape[0]
        if n != self.n or iteration % self.every == 0:
            self.update(gaussians)
        s = gaussians._scaling.data
        m = self.mask
        capped = torch.minimum(s[m], self.log_cap[m, None])
        self.n_clamped = int((capped < s[m]).any(1).sum())
        s[m] = capped

    def summary(self):
        return (f"fur Gaussians {int(self.mask.sum()) if self.mask is not None else 0}, "
                f"clamped at last step {self.n_clamped}, cap {self.cap_px:.2f} px")
