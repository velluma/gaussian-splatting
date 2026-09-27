#
# fur-gs: pseudo-view supervision for Difix3D+-style distillation (D25, method B1). Only active with --pseudo_dir.
# With the flag off, nothing here is imported or called and training is identical to vanilla 3DGS.
#
# <pseudo_dir>/pseudo_views.json lists cameras placed between neighbouring TRAIN cameras (never test poses):
#   {"views": [{"name", "image", "width", "height", "FoVx", "FoVy", "znear", "zfar",
#               "world_view_transform": 4x4, "full_proj_transform": 4x4}, ...]}
# and <pseudo_dir>/<image> is the Difix-fixed render used as the target. Built by tools/difix/b1_round.py.
# Each iteration, with probability --pseudo_prob a pseudo view replaces the training view; its loss
# (same L1 + SSIM as vanilla) is scaled by --pseudo_weight.
#
import json
import os
import random

import numpy as np
import torch
from PIL import Image


class PseudoCam:
    def __init__(self, v, root):
        self.image_name = v["name"]
        self.image_width, self.image_height = int(v["width"]), int(v["height"])
        self.FoVx, self.FoVy = float(v["FoVx"]), float(v["FoVy"])
        self.znear, self.zfar = float(v["znear"]), float(v["zfar"])
        self.world_view_transform = torch.tensor(v["world_view_transform"], dtype=torch.float32, device="cuda")
        self.full_proj_transform = torch.tensor(v["full_proj_transform"], dtype=torch.float32, device="cuda")
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]
        with Image.open(os.path.join(root, v["image"])) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # uint8 로 보관하고 쓸 때 float 로 바꾼다 (75장 1080p: float32 1.7 GiB -> uint8 0.43 GiB, D27)
        self._image_u8 = torch.from_numpy(a).permute(2, 0, 1).contiguous()
        self.alpha_mask = None
        self.depth_reliable = False
        self.invdepthmap = None
        self.depth_mask = None

    @property
    def original_image(self):
        return self._image_u8.float() / 255.0


class PseudoViews:
    def __init__(self, opt):
        root = opt.pseudo_dir
        meta = json.load(open(os.path.join(root, "pseudo_views.json"), encoding="utf-8"))
        self.cams = [PseudoCam(v, root) for v in meta["views"]]
        self.prob, self.weight = opt.pseudo_prob, opt.pseudo_weight
        self.n_used = 0
        print(f"[pseudo] {len(self.cams)} pseudo views from {root}; prob={self.prob} weight={self.weight}")

    def pick(self):
        if self.cams and random.random() < self.prob:
            self.n_used += 1
            return random.choice(self.cams)
        return None
