import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .renderer import render
from .cam_utils import orbit_camera, ortho_proj
from utils.logger import get_logger

log = get_logger(__name__)

class TextureDataset(Dataset):
    def __init__(self, texture_path, seen_views, target_res):
        self.texture_path = texture_path
        self.seen_views   = seen_views
        self.target_res   = target_res
    
    def __len__(self):
        return len(self.seen_views)
    
    def __getitem__(self, idx):
        yaw, pitch = self.seen_views[idx]
        image_pil = Image.open(os.path.join(self.texture_path, f'rgb_{yaw}_{pitch}.png')).resize((self.target_res, self.target_res))
        image_pt = torch.Tensor(np.array(image_pil) / 255.)[..., :-1].cuda()
        mask_pt = torch.Tensor(np.array(image_pil) / 255.)[..., -1][..., None].cuda().bool() 
        return image_pt, mask_pt, yaw, pitch

def texture_range_loss(A):
    loss = torch.pow(torch.where(A < 0.0, -A, torch.where(A > 1.0, A - 1, torch.zeros_like(A))), 2)
    return loss.sum()

def post_process(mesh, seen_views, texture_path, max_iter):
    mesh.albedo.requires_grad_()

    # Dataset and DataLoader
    target_res = 1024 * 2
    dataset = TextureDataset(texture_path, seen_views, target_res)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Optimizer setup
    optimizer  = torch.optim.Adam([mesh.albedo], lr=1e-2)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    regularization_weight = 1e+2 #1e-8

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    for it in range(max_iter):
        for batch_gt_images, batch_gt_masks, yaws, pitchs in loader:
            out_list = torch.empty((0, target_res, target_res, 3), device='cuda')
            for yaw, pitch in zip(yaws, pitchs):
                pose = orbit_camera(180 + pitch, yaw, opengl=True)
                proj = ortho_proj()
                out = render(mesh, pose, proj, h=target_res, w=target_res)
                out_list = torch.cat([out_list, out['image_no_aa'][..., :-1]])

            total_loss = 0.
            l1_loss  = L1_loss(out_list[batch_gt_masks.expand(-1, -1, -1, 3)], batch_gt_images[batch_gt_masks.expand(-1, -1, -1, 3)])
            mse_loss = MSE_loss(out_list[batch_gt_masks.expand(-1, -1, -1, 3)], batch_gt_images[batch_gt_masks.expand(-1, -1, -1, 3)])
            tr_loss  = regularization_weight * texture_range_loss(mesh.albedo)
            total_loss = l1_loss + tr_loss
            #total_loss = l1_loss + mse_loss + tr_loss

            #print(f'iter = {it}, total_loss = {total_loss}, l1_loss = {l1_loss}, tr_loss = {tr_loss}')
            log.info(f'iter = {it}, total_loss = {total_loss}, l1_loss = {l1_loss}, mse_loss = {mse_loss}, tr_loss = {tr_loss}')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        scheduler.step()

    return mesh
