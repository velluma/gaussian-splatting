#
# fur-gs: fur-region oriented spectral texture loss (D23). Only active with --tex_loss.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# For each training view, pick random PxP patches that lie inside the fur interior found by the RGB-only
# detector (tools/fur_detect). For rendered and GT patches, compute the phase-free power spectrum
# (per RGB channel, mean removed, Hann window), average it into (radius x orientation) bins and compare in log scale:
#     L_tex = mean | log(P_render + f) - log(P_gt + f) |,  f = tex_floor * P_gt + eps
# Dropping the phase makes the loss insensitive to where strands are; the orientation bins keep the
# direction of the fur texture. The floor f bounds the penalty for missing power (blurry early renders),
# which otherwise dominates training. Total loss = original 3DGS loss + tex_lambda * L_tex.
# By default the texture gradient is kept out of the densification statistics (train.py, --tex_densify off),
# so densification follows the original pixel-loss rule and the texture term only shapes the Gaussians.
#
import math
import os

import numpy as np
import torch
from PIL import Image


def _load_gray(path, size):
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != size:
            im = im.resize(size, Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)


class TexLoss:
    def __init__(self, opt, dataset, train_cameras):
        self.lam = opt.tex_lambda
        self.P = opt.tex_patch
        self.n = opt.tex_npatch
        self.from_iter = opt.tex_from_iter
        self.eps = opt.tex_eps
        self.floor = opt.tex_floor
        map_dir = opt.tex_maps or opt.fur_maps or os.path.join(dataset.source_path, "fur_maps")
        if not os.path.isdir(map_dir):
            raise FileNotFoundError(f"[tex] fur map folder not found: {map_dir} (run tools/fur_detect/detect.py)")
        P, stride = self.P, max(self.P // 4, 1)
        self.pos = {}
        n_total = 0
        for cam in train_cameras:
            stem = os.path.splitext(cam.image_name)[0]
            m = _load_gray(os.path.join(map_dir, f"{stem}_interior.png"), (cam.image_width, cam.image_height))
            fur = (m > opt.tex_min_score).float()
            ii = torch.nn.functional.pad(fur.cumsum(0).cumsum(1), (1, 0, 1, 0))
            H, W = fur.shape
            ys = torch.arange(0, H - P + 1, stride)
            xs = torch.arange(0, W - P + 1, stride)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            cov = (ii[yy + P, xx + P] - ii[yy, xx + P] - ii[yy + P, xx] + ii[yy, xx]) / (P * P)
            ok = cov >= opt.tex_min_cover
            self.pos[cam.image_name] = torch.stack([yy[ok], xx[ok]], 1).cuda()
            n_total += int(ok.sum())
        # Hann window and (radius x orientation) bins of the rfft2 grid
        h = torch.hann_window(P, periodic=False)
        self.window = (h[:, None] * h[None, :]).cuda()
        fy = torch.fft.fftfreq(P)[:, None].expand(P, P // 2 + 1)
        fx = torch.fft.rfftfreq(P)[None, :].expand(P, P // 2 + 1)
        r = torch.sqrt(fx ** 2 + fy ** 2)
        ang = torch.remainder(torch.atan2(fy, fx), math.pi)                  # orientation 0..pi
        nr, na = opt.tex_rbins, opt.tex_abins
        rb = torch.clamp((r / 0.5 * nr).long(), max=nr)                      # nr = beyond Nyquist (corners)
        ab = torch.clamp((ang / math.pi * na).long(), max=na - 1)
        valid = (r > 0) & (rb < nr)                                          # drop DC and corners
        idx = torch.where(valid, rb * na + ab, torch.full_like(rb, nr * na))
        self.nbins = nr * na
        self.idx = idx.flatten().cuda()
        cnt = torch.zeros(self.nbins + 1).index_add_(0, idx.flatten(), torch.ones(idx.numel()))
        self.cnt = cnt[:self.nbins].clamp(min=1).cuda()
        self.ema = None
        print(f"[tex] maps from {map_dir}: {n_total} candidate {P}x{P} patches over {len(self.pos)} train views "
              f"(interior>{opt.tex_min_score}, cover>={opt.tex_min_cover}); lambda={self.lam} npatch={self.n} "
              f"bins={nr}x{na} from_iter={self.from_iter}")

    def _binned_power(self, x):
        """(N,3,P,P) -> (N*3, nbins) mean power per (radius, orientation) bin, per RGB channel
        (per channel so that the loss cannot be met with colour-shifted lines; see D23 pilot)."""
        g = x.reshape(-1, x.shape[-2], x.shape[-1])
        g = (g - g.mean((1, 2), keepdim=True)) * self.window
        pw = torch.fft.rfft2(g).abs() ** 2                                  # (N,P,P/2+1)
        out = torch.zeros(pw.shape[0], self.nbins + 1, device=x.device, dtype=pw.dtype)
        out.index_add_(1, self.idx, pw.flatten(1))
        return out[:, :self.nbins] / self.cnt

    def __call__(self, iteration, image, gt, cam):
        if iteration < self.from_iter:
            return None
        pos = self.pos.get(cam.image_name)
        if pos is None or len(pos) == 0:
            return None
        pick = pos[torch.randint(len(pos), (self.n,), device=pos.device)]
        P = self.P
        r = torch.stack([image[:, y:y + P, x:x + P] for y, x in pick.tolist()])
        g = torch.stack([gt[:, y:y + P, x:x + P] for y, x in pick.tolist()])
        pr = self._binned_power(r)
        with torch.no_grad():
            pg = self._binned_power(g)
            floor = self.floor * pg + self.eps        # bounded penalty for missing power: log(1 + 1/floor)
        loss = (torch.log(pr + floor) - torch.log(pg + floor)).abs().mean()
        v = float(loss.detach())
        self.ema = v if self.ema is None else 0.99 * self.ema + 0.01 * v
        return self.lam * loss
