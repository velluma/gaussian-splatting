#
# fur-gs: external comparison method "effective rank regularization" (docs/erank.md).
# Hyung et al., Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting, NeurIPS 2024
# (arXiv 2406.11672, official code github.com/junhahyung/erank_gs). Only active with --erank_lambda > 0.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# For each Gaussian with scales s (3 axes), q_i = s_i^2 / sum_j s_j^2 and
#     erank = exp(H(q)),  H(q) = -sum_i q_i log q_i            (1 = needle, 2 = disk, 3 = sphere)
#     L = erank_lambda * mean_k max(-log(erank_k - 1 + eps), 0) + erank_thin_lambda * mean_k min_i s_ik
# applied from --erank_from_iter (paper: 7000) to the end. Paper eq. 10 writes a sum over Gaussians and an unweighted
# smallest-scale term; the official code takes the mean over Gaussians and weights the smallest-scale term with
# thin_lambda = 1 (train.sh), which is what we use. eps: paper 1e-5 (official code 1e-7).
# The paper's "+e" also changes densification (GOF-style abs-gradient ADC, needs a modified rasterizer); NOT included here.
# The loss depends only on the scales, so the view-space gradients used for densification are unchanged.
#
import torch


def effective_rank(scales, eps=1e-12):
    """scales: (N,3) positive axis lengths -> (N,) effective rank in [1,3]."""
    d = scales * scales
    q = d / d.sum(dim=1, keepdim=True)
    h = -(q * torch.log(q.clamp_min(eps))).sum(dim=1)
    return torch.exp(h)


def erank_loss(scales, lam, thin_lam, eps=1e-5):
    """Returns (loss, erank). loss = lam * mean(relu(-log(erank - 1 + eps))) + thin_lam * mean(min scale)."""
    er = effective_rank(scales)
    l_rank = torch.clamp(-torch.log(er - 1.0 + eps), min=0.0).mean()
    l_thin = scales.min(dim=1).values.mean()
    return lam * l_rank + thin_lam * l_thin, er


class ErankReg:
    def __init__(self, opt):
        self.lam = opt.erank_lambda
        self.thin_lam = opt.erank_thin_lambda
        self.from_iter = opt.erank_from_iter
        self.eps = opt.erank_eps
        print(f"[erank] lambda {self.lam}, thin lambda {self.thin_lam}, eps {self.eps}, from iter {self.from_iter}")

    def __call__(self, iteration, gaussians):
        if iteration < self.from_iter:
            return None
        loss, er = erank_loss(gaussians.get_scaling, self.lam, self.thin_lam, self.eps)
        if iteration % 1000 == 0:
            er = er.detach()
            print(f"\n[erank] ITER {iteration}: loss {loss.item():.5f}, mean erank {er.mean().item():.3f}, "
                  f"erank<1.5 {(er < 1.5).float().mean().item():.3f}, gaussians {er.shape[0]}")
        return loss
