#
# fur-gs: WD-R perceptual loss (docs/wdr.md). Only active with --wd_loss.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# Ozyilkan et al., "Drop-In Perceptual Optimization for 3D Gaussian Splatting", ECCV 2026 (arXiv 2603.23297):
#     L_WD-R = gamma * (d_WD + beta * L_orig),  L_orig = (1 - lambda_dssim) L1 + lambda_dssim (1 - SSIM)
#     constant pooling sigma = 4 px, original loss only for the first 3k iterations (warm-up), beta = 1/0.09 in the paper.
# Wasserstein distortion (Qiu et al. 2023; implementation described in Balle et al. 2024, arXiv 2412.00505):
#     for every feature map (image pixels as the 0th feature + VGG activations) local means mu and standard deviations nu
#     are taken from a 2x Gaussian pyramid of the first and second moments (3x3 binomial filter, stride 2);
#     level a pools about 2^a feature pixels. sigma selects levels by w_a = max(1 - |log2(sigma / stride) - a|, 0).
#     d = sqrt((mu - mu')^2 + (nu - nu')^2), averaged over locations and channels, summed over levels, averaged over features.
# No official code was released. Our choices (docs/wdr.md):
#   - VGG16 (torchvision ImageNet weights, the same file LPIPS uses), features relu1_2 relu2_2 relu3_3 relu4_3 relu5_3 + pixels.
#     Layers whose stride is >= sigma (relu3_3 onward at sigma 4) are compared pointwise (level 0).
#   - d_WD is computed on the object box of the training photo (non-white pixels + margin, from the RGB photo only, D3);
#     L_orig stays on the full image as in vanilla 3DGS.
#   - beta and gamma are set automatically (unless given) from image-space gradient norms measured on the first
#     --wd_calib_iters iterations after warm-up, while training still uses the original loss:
#       beta : ||grad d_WD|| / ||grad beta*L_orig|| = --wd_grad_ratio (paper reports a mean ratio of ~1.6)
#       gamma: ||grad gamma*(d_WD + beta*L_orig)|| = ||grad L_orig||  (keeps the gradient scale that drives densification;
#              the paper tunes gamma per dataset to keep splat counts comparable)
#     The values are written to <model_path>/wd_calib.json and used for the rest of training.
#
import json
import math
import os

import torch
import torch.nn.functional as F

VGG_LAYERS = {3: 1, 8: 2, 15: 4, 22: 8, 29: 16}          # torchvision vgg16.features index of relu1_2 ... relu5_3 -> stride
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def level_weights(sigma, stride):
    """{level: weight} for pooling width sigma (image px) on a feature map with the given stride."""
    l = math.log2(sigma / stride) if sigma > 0 else -1.0
    if l <= 0:
        return {0: 1.0}
    a0 = int(math.floor(l))
    w = {a0: 1.0 - (l - a0), a0 + 1: l - a0}
    return {a: v for a, v in w.items() if v > 1e-6}


def _down(x):
    """3x3 binomial blur + stride 2 (depthwise), reflect padding."""
    c = x.shape[1]
    k = torch.tensor([1.0, 2.0, 1.0], device=x.device, dtype=x.dtype)
    k = (k[:, None] * k[None, :]) / 16.0
    k = k.expand(c, 1, 3, 3).contiguous()
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, stride=2, groups=c)


def moment_pyramid(f, max_level):
    """[(E[f], E[f^2])] for levels 0..max_level."""
    m, q = f, f * f
    out = [(m, q)]
    for _ in range(max_level):
        if min(m.shape[-2:]) < 3:
            break
        m, q = _down(m), _down(q)
        out.append((m, q))
    return out


class WDLoss:
    def __init__(self, opt, model_path):
        import torchvision
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:max(VGG_LAYERS) + 1]
        self.vgg = vgg.cuda().eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.mean, self.std = IMNET_MEAN.cuda(), IMNET_STD.cuda()
        self.sigma = opt.wd_sigma
        self.from_iter = opt.wd_from_iter
        self.ratio = opt.wd_grad_ratio
        self.calib_iters = opt.wd_calib_iters
        self.margin = opt.wd_margin
        self.max_px = opt.wd_max_px
        self.eps = 1e-6
        self.beta = opt.wd_beta if opt.wd_beta >= 0 else None
        self.gamma = opt.wd_gamma if opt.wd_gamma > 0 else None
        self.model_path = model_path
        self.boxes = {}
        self.calib = []                                   # per iteration (|g_wd|^2, <g_wd, g_or>, |g_or|^2)
        self.feat_levels = [(1, level_weights(self.sigma, 1))] + [(s, level_weights(self.sigma, s)) for s in VGG_LAYERS.values()]
        self.ema_wd = None
        print(f"[wd] sigma {self.sigma}, from iter {self.from_iter}, beta {self.beta if self.beta is not None else 'auto'}, "
              f"gamma {self.gamma if self.gamma is not None else 'auto'} (grad ratio {self.ratio}, {self.calib_iters} calibration iters), "
              f"levels {[w for _, w in self.feat_levels]}")

    # ------------------------------------------------------------------ features
    def features(self, img):
        """img (3,H,W) in [0,1] -> [pixels, relu1_2, relu2_2, relu3_3, relu4_3, relu5_3], each (1,C,h,w)."""
        x = img.unsqueeze(0)
        feats = [x]
        h = (x - self.mean) / self.std
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in VGG_LAYERS:
                feats.append(h)
        return feats

    def box(self, cam, gt):
        """Object box of the training photo: pixels that are not (near) white, + margin. RGB photo only (no alpha, D3)."""
        key = cam.image_name
        if key not in self.boxes:
            with torch.no_grad():
                fg = (gt < 0.98).any(dim=0)
                H, W = fg.shape
                if fg.any():
                    ys = torch.nonzero(fg.any(dim=1)).flatten()
                    xs = torch.nonzero(fg.any(dim=0)).flatten()
                    y0, y1 = max(int(ys[0]) - self.margin, 0), min(int(ys[-1]) + 1 + self.margin, H)
                    x0, x1 = max(int(xs[0]) - self.margin, 0), min(int(xs[-1]) + 1 + self.margin, W)
                else:
                    y0, y1, x0, x1 = 0, H, 0, W
            self.boxes[key] = (y0, y1, x0, x1)
        y0, y1, x0, x1 = self.boxes[key]
        # very large boxes: a random window of at most max_px (keeps VGG memory bounded)
        h, w = y1 - y0, x1 - x0
        if h * w > self.max_px:
            s = math.sqrt(self.max_px / (h * w))
            ch, cw = max(int(h * s), 64), max(int(w * s), 64)
            oy = int(torch.randint(0, h - ch + 1, (1,)))
            ox = int(torch.randint(0, w - cw + 1, (1,)))
            y0, x0, y1, x1 = y0 + oy, x0 + ox, y0 + oy + ch, x0 + ox + cw
        return y0, y1, x0, x1

    def distance(self, image, gt, cam):
        y0, y1, x0, x1 = self.box(cam, gt)
        fr = self.features(image[:, y0:y1, x0:x1])
        with torch.no_grad():
            fg = self.features(gt[:, y0:y1, x0:x1])
        total = 0.0
        for (stride, wts), a, b in zip(self.feat_levels, fr, fg):
            top = max(wts)
            pa, pb = moment_pyramid(a, top), moment_pyramid(b, top)
            for lev, w in wts.items():
                if lev >= len(pa):
                    lev = len(pa) - 1
                (ma, qa), (mb, qb) = pa[lev], pb[lev]
                if lev == 0:
                    d = torch.sqrt((ma - mb) ** 2 + self.eps)
                else:
                    na = torch.sqrt(torch.clamp(qa - ma * ma, min=0.0) + self.eps)
                    nb = torch.sqrt(torch.clamp(qb - mb * mb, min=0.0) + self.eps)
                    d = torch.sqrt((ma - mb) ** 2 + (na - nb) ** 2 + self.eps)
                total = total + w * d.mean()
        return total / len(fr)

    # ------------------------------------------------------------------ loss
    def __call__(self, iteration, image, gt, cam, loss_orig):
        """Returns the loss to back-propagate (loss_orig before the WD phase / during calibration)."""
        if iteration < self.from_iter:
            return loss_orig
        dwd = self.distance(image, gt, cam)
        self.ema_wd = dwd.item() if self.ema_wd is None else 0.9 * self.ema_wd + 0.1 * dwd.item()
        if self.beta is None or self.gamma is None:
            if len(self.calib) < self.calib_iters:
                g_wd = torch.autograd.grad(dwd, image, retain_graph=True)[0]
                g_or = torch.autograd.grad(loss_orig, image, retain_graph=True)[0]
                self.calib.append(((g_wd * g_wd).sum().item(), (g_wd * g_or).sum().item(), (g_or * g_or).sum().item()))
                if len(self.calib) == self.calib_iters:
                    self._finish_calibration(iteration)
                return loss_orig                           # training continues with the original loss while measuring
        return self.gamma * (dwd + self.beta * loss_orig)

    def _finish_calibration(self, iteration):
        a = [c[0] for c in self.calib]
        b = [c[1] for c in self.calib]
        c = [c[2] for c in self.calib]
        n_wd = sum(math.sqrt(x) for x in a) / len(a)
        n_or = sum(math.sqrt(x) for x in c) / len(c)
        if self.beta is None:
            self.beta = n_wd / (self.ratio * n_or)
        n_tot = sum(math.sqrt(max(ai + 2 * self.beta * bi + self.beta ** 2 * ci, 0.0)) for ai, bi, ci in zip(a, b, c)) / len(a)
        if self.gamma is None:
            self.gamma = n_or / n_tot
        info = {"iteration": iteration, "calib_iters": len(self.calib), "sigma": self.sigma, "grad_ratio_target": self.ratio,
                "mean_grad_norm_wd": n_wd, "mean_grad_norm_orig": n_or, "mean_grad_norm_combined_unscaled": n_tot,
                "beta": self.beta, "gamma": self.gamma,
                "effective_weights": {"d_WD": self.gamma, "L_orig": self.gamma * self.beta}}
        print(f"\n[wd] calibration done at iter {iteration}: beta {self.beta:.4g}, gamma {self.gamma:.4g} "
              f"(|g_wd| {n_wd:.4g}, |g_orig| {n_or:.4g}) -> loss = {self.gamma:.4g} d_WD + {self.gamma * self.beta:.4g} L_orig")
        with open(os.path.join(self.model_path, "wd_calib.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
