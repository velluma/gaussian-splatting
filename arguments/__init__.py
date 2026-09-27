#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.split_file = ""                # fur-gs (D28): JSON {"train": [...], "test": [...]} image names; "" = 3DGS llffhold rule
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # fur-gs (D14, D15). All off by default -> vanilla 3DGS
        self.fur_densify = False            # lower the densify threshold on fur Gaussians (tools/fur_detect maps)
        self.fur_maps = ""                  # default: <source_path>/fur_maps
        self.fur_mode = "both"              # both | fringe | interior : which fur component drives densification
        self.fur_beta = 1.0                 # grad multiplier = 1 + beta * fur score
        self.fur_decay = 0.8                # forgetting factor of fur statistics per densification step
        self.fur_orient_split = False       # split fur-interior Gaussians along the 3D strand direction
        self.fur_orient_clone = False       # also orient clones of fur-interior Gaussians along the strand
        self.fur_orient_aspect = 0.5        # across-strand / along-strand scale ratio (upper bound) of children
        self.fur_orient_min_conf = 0.5      # min direction confidence (1 - l0/l1)
        self.fur_orient_min_interior = 0.3  # min mean interior fur score
        # fur-gs (D23). Off by default -> vanilla 3DGS loss
        self.tex_loss = False               # add lambda * oriented spectral texture loss on fur-interior patches
        self.tex_lambda = 0.1
        self.tex_maps = ""                  # default: --fur_maps, else <source_path>/fur_maps
        self.tex_patch = 64                 # patch size (px)
        self.tex_npatch = 8                 # patches per iteration
        self.tex_min_score = 0.2            # detector interior score that counts as fur
        self.tex_min_cover = 0.9            # min fraction of fur pixels in a patch
        self.tex_rbins = 8                  # radial frequency bins (0..0.5 cycles/px)
        self.tex_abins = 8                  # orientation bins (0..pi)
        self.tex_eps = 1e-6                 # log(power + floor * gt_power + eps)
        self.tex_floor = 0.1                # bounds the penalty for missing power to log(1 + 1/floor)
        self.tex_from_iter = 1000
        self.tex_densify = False            # let the texture gradient also drive densification (off: pixel loss only)
        # fur-gs (D25, method B1). Off by default -> vanilla 3DGS
        self.pseudo_dir = ""                # folder with pseudo_views.json + Difix-fixed pseudo targets
        self.pseudo_prob = 0.3              # probability that an iteration uses a pseudo view instead of a train view
        self.pseudo_weight = 1.0            # loss weight of pseudo-view iterations
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
