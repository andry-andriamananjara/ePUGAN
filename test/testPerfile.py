import argparse
import os, sys
sys.path.append("../")

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--resume', type=str, required=True)
parser.add_argument('--exp_name',type=str,required=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.xyz_util import save_xyz_file

from network.networks import Generator
from data.data_loader import PUNET_Dataset_Whole


import numpy as np
from data.data_loader import PUNET_Dataset
import importlib
from network.networks import Generator
from option.train_option import get_train_options
from pytorch3d.ops.utils import masked_gather
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.utils import masked_gather
from pytorch3d.loss.chamfer import chamfer_distance
import point_cloud_utils as pcu
import math

NUM_POINT = 312

def get_cd_loss(pred, gt, pcd_radius):
    #idx, _ = auction_match(pred, gt)
    #matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    #matched_out = matched_out.transpose(1, 2).contiguous()
    #dist2 = (pred - matched_out) ** 2
    #dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
    #dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
    #dist2 /= pcd_radius
    #return torch.mean(dist2)
    
    '''
    pred and gt is (N, P1, D)
    '''
    #print('Pred :: ', pred.shape)
    #print('GT :: ',gt.shape)
    cham_x,_ = chamfer_distance(pred, gt,batch_reduction = "mean", point_reduction = "mean", norm = 2)
    #print('Shape cham_x :: ',cham_x.shape)

    return cham_x

def pc_normalization(pc):
    
    centroid = torch.mean(pc[..., :3], dim=1, keepdims=True)
    furthest_distance = torch.max(torch.norm(pc[...,:3]-centroid, dim=1))
    radius   = furthest_distance  # not very sure?
    
    #print('PC         :: \n',pc[..., :3])
    #print('Centroid   :: \n',centroid)
    #print('Difference ::\n', pc[...,:3]-centroid)
    #print('Norm       :: \n',torch.norm(pc[...,:3]-centroid, dim=1))
    
    #normalization
    pc     = (pc-centroid)/radius
    
    return pc, centroid, radius

#def pc_prediction(net, input_pc, patch_num_ratio=3):
def pc_prediction(input_pc, patch_num_ratio=3):
    """
    upsample patches of a point cloud
    :param
        input_pc        1x3xN
        patch_num_ratio int, impacts number of patches and overlapping
    :return
        input_list      list of [3xM]
        up_point_list   list of [3xMr]
    """
    
    # divide to patches
    seeds_num = int(input_pc.shape[1] / NUM_POINT * patch_num_ratio)
    
    # FPS sampling
    start = time.time()
    pc_seeds, seeds_idx  = sample_farthest_points(input_pc.contiguous(), K=seeds_num, random_start_point=False)    
    
    # Grouping
    _, idx, patches = knn_points(pc_seeds, input_pc, K = NUM_POINT, return_nn = True,return_sorted = True)
    patches = patches.permute(0,2,1,3)
    
    print('Input           :: ', input_pc.shape)
    print('Number of Seeds :: ', seeds_num)
    print('Seed Point      :: ', pc_seeds.shape)
    print('Seed Idx list   :: ', seeds_idx.shape)
    print('Patches shape   :: ', patches.shape)
    
    input_list = []
    up_point_list = []

    for k in tqdm(range(seeds_num)):
        patch = patches[:, :, k, :]
        patch, centroid, radius =pc_normalization(patch)
    #    up_point = net.forward(patch.detach(), ratio=UP_RATIO)
    #    up_point = up_point * radius + centroid
    #    input_list.append(patch)
    #    up_point_list.append(up_point)

    #return input_list, up_point_list

if __name__ == '__main__':

#///////////////////////
#///////////////////////

    model = Generator()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    #eval_dst = PUNET_Dataset_Whole(data_dir='../MC_5k')
    eval_dst = PUNET_Dataset_Whole(data_dir='../MC_5k/GT')
    #eval_dst = PUNET_Dataset_Whole(data_dir='../MC_5k/Test_PUNET')

    eval_loader = DataLoader(eval_dst, batch_size=1,
                             shuffle=False, pin_memory=True, num_workers=0)

    emd_list = []
    cd_list  = []
    hd_list  = []

    with torch.no_grad():
        for itr, batch in enumerate(eval_loader):
            points, gt, radius = batch

            #print('Inputs :: ',points.shape,' GT :: ', gt.shape)
            points = points[:,:,:3].permute(0,2,1).float().cuda().contiguous()
            gt = gt[:,:,:3].float().cuda().contiguous()
            radius = radius.float().cuda()
            preds  = model(points)  # points.shape[1])
            preds  = preds.permute(0,2,1).contiguous()
            
            #print('Preds :: ',preds.shape,' GT :: ', gt.shape)
            #emd = get_emd_loss(preds, gt, radius)
            
            
            a = torch.cat(torch.unbind(preds,dim=1),dim=0).data.cpu().numpy() #preds.data.cpu().numpy()
            b = torch.cat(torch.unbind(gt,dim=1),dim=0).data.cpu().numpy() #gt.data.cpu().numpy()
            hd = pcu.hausdorff_distance(a, b)
            cd = get_cd_loss(preds, gt, radius)

            #print(' -- iter {}, emd {}, cd {}.'.format(itr, emd, cd))
            print(' -- iter {}, input {}, pred {}, gt {}, cd {}, hd {}.'.format(itr, points.shape[2], preds.shape[1], gt.shape[2], cd, hd))
            #emd_list.append(emd.item())
            cd_list.append(cd.item())
            hd_list.append(hd)

    #print('mean emd     : {}'.format(np.mean(emd_list)))
    print('mean cd       : {}'.format(np.mean(cd_list)))
    print('mean hd       : {}'.format(np.mean(hd_list)))
