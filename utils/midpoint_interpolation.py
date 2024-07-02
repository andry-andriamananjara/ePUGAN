'''
@InProceedings{He_2023_CVPR,
    author    = {He, Yun and Tang, Danhang and Zhang, Yinda and Xue, Xiangyang and Fu, Yanwei},
    title     = {Grad-PU: Arbitrary-Scale Point Cloud Upsampling via Gradient Descent with Learned Distance Functions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
'''

import os,sys
sys.path.append("../")
import torch
import numpy as np
import random
from einops import rearrange, repeat
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points

import utils.data_util as utils
from data.data_loader import PUNET_Dataset, PUNET_PU1K_Dataset
import cv2
import numpy as np
from utils.pc_util import draw_point_cloud
from data.data_loader import PUNET_Dataset, PUNET_PU1K_Dataset
from torch.utils import data

def plot_save_pcd(pcd, file, exp_name):

    image_save_dir=os.path.join("viz_sample",exp_name)

    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    #file_name=file.split("/")[-1].split('.')[0]
    file_name=file
    
    img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                           diameter=4)
    img=(img*255).astype(np.uint8)
    image_save_path=os.path.join(image_save_dir,file_name+".png")
    cv2.imwrite(image_save_path,img)

def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    
    # (b, m, k)
    #knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    _,knn_idx, knn_pts = knn_points(pts_trans, center_pts_trans,K = k, return_nn = True,return_sorted = True)
    
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def midpoint_interpolate(sparse_pts, up_rate=4):
    # sparse_pts: (b, 3, n)
    
    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts.permute(0,2,1) #b,3,N
    
    # fps: b,N,3
    interpolated_pts, _   = sample_farthest_points(interpolated_pts.contiguous(), K=up_pts_num, random_start_point=False)
    interpolated_pts      = interpolated_pts.permute(0,2,1) #b,3,N

    return interpolated_pts


def MI_prediction(input_data, alpha):
    '''
     input: bx3xN
    '''
    #print('xxx MI input : ',input_data.shape)
    sparse_pts=input_data
    #sparse_pts        = sparse_pts.permute(0,2,1)
    MI_pc             = midpoint_interpolate(sparse_pts, up_rate=alpha)
    #print('xxx MI output : ', MI_pc.shape)
    #MI_pc             = MI_pc.permute(0,2,1)
    return MI_pc
    
if __name__=="__main__":
    
    uniform_state = True
    date_name     = 'pu1k_scale'
    exp_name      = 'mid_viz'
    dataset=PUNET_PU1K_Dataset(uniform=uniform_state, dataname = date_name)
    
    for batch_id in range(3):
        input_data,gt_data,radius_data=dataset.__getitem__(batch_id)
    
        print('xxGT    : ',gt_data.shape)
        print('xxInput : ',input_data.shape)
        data_npoint = gt_data.shape[0]
        npoint      = gt_data.shape[0]
        alpha       = 4
        print('xxAlpha : ',alpha)
        if not uniform_state:
            sample_idx = utils.nonuniform_sampling(data_npoint, sample_num=int(npoint/alpha))
            input_data = input_data[sample_idx, :]
            input_data = torch.from_numpy(input_data)
            input_data = input_data[np.newaxis,...]
            file       = 'non_uniform_'+str(batch_id)+'_input'
        else:
            input_data = torch.from_numpy(input_data)
            input_data = input_data[np.newaxis,...]
            input_data, _   = sample_farthest_points(input_data.contiguous(), K=int(npoint/alpha), random_start_point=False)
            file       = 'uniform_'+str(batch_id)+'_input'
        
        # Prediction
        
        MI_pc             = MI_prediction(input_data, alpha)
        input_data = input_data[0, ...]
        input_data = input_data.detach().cpu().numpy() 
        MI_pc      = MI_pc[0, ...]
        MI_pc      = MI_pc.detach().cpu().numpy()

        print('xxInput   : ',input_data.shape)
        print('xxPredict : ',MI_pc.shape)
        plot_save_pcd(input_data, file, exp_name)
        plot_save_pcd(MI_pc, file+'_MI', exp_name)
        plot_save_pcd(gt_data, file+'_GT', exp_name)

    #sparse_pts = torch.rand(4,3,8192)#.cuda()
    #print(sparse_pts.shape)
    #up_rate_list = [2, 4, 8, 9]
    #sparse_pts        = sparse_pts.permute(0,2,1)
    #for i in up_rate_list:
    #    sparse_scale, _   = sample_farthest_points(sparse_pts.contiguous(), K=int(sparse_pts.shape[1]/i), random_start_point=False)
    #    print(sparse_scale.shape)
    #    sparse_scale      = sparse_scale.permute(0,2,1)
    #    MI_pc             = midpoint_interpolate(sparse_scale, up_rate=i)
    #    MI_pc             = MI_pc.permute(0,2,1)
    #    print(MI_pc.shape)
    