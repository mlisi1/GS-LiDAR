#
# Export a 3D point cloud (.ply) from a training checkpoint, for a given
# iteration, so you can inspect the actual reconstructed geometry in
# Open3D/CloudCompare instead of only the 2D depth/intensity panorama PNGs
# that complete_eval() writes during training.
#
# This mirrors train.py's own checkpoint-restore path (see `training()`)
# and reuses GS-LiDAR's own render_range_map + pano_to_lidar + save_ply --
# no new projection math, just wiring the existing pieces together for a
# single checkpoint instead of the full training loop.
#
# Usage:
#   python scripts/export_ply.py --config configs/vlp16_test_seq_0000.yaml \
#       source_path=<data dir> model_path=eval_output/vlp16_test \
#       --iteration 7000 --out eval_output/vlp16_test/ply_7000
#
# Writes, per validation frame: pred_XXX.ply (reconstructed, raydrop-masked,
# intensity-colored) and gt_XXX.ply (ground truth, for side-by-side
# comparison in the same viewer).
import argparse
import os
import sys

# gaussian_renderer/scene/utils are top-level packages at the repo root, but
# this script lives in scripts/ -- running `python scripts/export_ply.py`
# only puts scripts/ on sys.path, not the repo root, so add it explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from gaussian_renderer import render, render_range_map
from scene import Scene, GaussianModel, RayDropPrior
from utils.graphics_utils import pano_to_lidar
from utils.system_utils import save_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--iteration", type=int, required=True,
                         help="Checkpoint iteration to load (must have a matching ckpt/chkpnt<N>.pth).")
    parser.add_argument("--out", type=str, required=True, help="Output directory for the .ply files.")
    parser.add_argument("--frame-idx", type=int, default=None,
                         help="Export only this validation-frame index (0-based, in val_frame_ids order). Default: export all.")
    args_read, unknown = parser.parse_known_args()

    base_conf = OmegaConf.load(args_read.base_config)
    second_conf = OmegaConf.load(args_read.config)
    cli_conf = OmegaConf.from_cli(unknown)
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)

    args_read.out = os.path.abspath(args_read.out)
    os.makedirs(args_read.out, exist_ok=True)
    print(f"Writing .ply files to: {args_read.out}")

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)

    ckpt_path = os.path.join(args.model_path, "ckpt", f"chkpnt{args_read.iteration}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- check --iteration and model_path.")
    model_params, saved_iteration = torch.load(ckpt_path, weights_only=False)
    gaussians.restore(model_params, args)

    raydrop_ckpt_path = os.path.join(os.path.dirname(ckpt_path),
                                      os.path.basename(ckpt_path).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
    start_w, start_h = scene.getWH()
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    raydrop_params, _ = torch.load(raydrop_ckpt_path, weights_only=False)
    lidar_raydrop_prior.restore(raydrop_params)

    # Match the resolution scale this checkpoint was actually trained at
    # (scene.upScale() is what train.py calls when resuming -- see training()).
    for _ in range(saved_iteration // args.scale_increase_interval):
        scene.upScale()

    bg_color = [1, 1, 1, 1] if args.white_background else [0, 0, 0, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scale = scene.resolution_scales[scene.scale_index]
    h, w = args.hw
    h //= scale
    w //= scale

    test_cameras = scene.getTestCameras(scale=scale)
    n_pairs = len(test_cameras) // 2
    frame_range = [args_read.frame_idx] if args_read.frame_idx is not None else range(n_pairs)

    with torch.no_grad():
        for idx in frame_range:
            cam_front = test_cameras[idx * 2]
            cam_back = test_cameras[idx * 2 + 1]

            depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano = render_range_map(
                args, cam_front, cam_back, scene.gaussians, render, (args, background), lidar_raydrop_prior, [h, w]
            )

            raydrop_mask = torch.where(raydrop_pano > 0.5, 1, 0)
            gt_raydrop_mask = torch.where(gt_depth_pano > 0, 0, 1)

            # depth_pano channel 0 is the "mix" depth (var-gated blend of
            # mean/median) -- the same channel complete_eval() reports its
            # headline Depth metrics against (see train.py:complete_eval).
            pred_depth = depth_pano[[0]] * (1.0 - raydrop_mask)
            pred_intensity = intensity_sh_pano * (1.0 - raydrop_mask)

            pred_xyz = pano_to_lidar(pred_depth, args.vfov, (-180, 180))
            gt_xyz = pano_to_lidar(gt_depth_pano, args.vfov, (-180, 180))

            mask = pred_depth[0] > 0
            pred_intensity_flat = pred_intensity[0][mask].unsqueeze(-1)
            gt_mask = gt_depth_pano[0] > 0
            gt_intensity_flat = gt_intensity_pano[0][gt_mask].unsqueeze(-1)

            frame_id = cam_front.colmap_id
            save_ply(pred_xyz, os.path.join(args_read.out, f"pred_{frame_id:03d}.ply"), rgbs=pred_intensity_flat)
            save_ply(gt_xyz, os.path.join(args_read.out, f"gt_{frame_id:03d}.ply"), rgbs=gt_intensity_flat)
            print(f"wrote frame {frame_id}: {pred_xyz.shape[0]} pred points, {gt_xyz.shape[0]} gt points")


if __name__ == "__main__":
    main()
