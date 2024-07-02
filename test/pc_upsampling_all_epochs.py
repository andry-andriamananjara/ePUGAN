"""
    This file is inspired by the upsampling techniques in 3PU
    @InProceedings{Yifan_2019_CVPR,
    author = {Yifan, Wang and Wu, Shihao and Huang, Hui and Cohen-Or, Daniel and Sorkine-Hornung, Olga},
    title = {Patch-Based Progressive 3D Point Set Upsampling},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    }
"""
import argparse
import os, sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
import time
import open3d as o3d
import glob 
import cv2
import utils.data_util as utils
import h5py
import math
import pandas as pd
import point_cloud_utils as pcu
import random

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.sample_farthest_points import sample_farthest_points_naive
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.utils import masked_gather
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.point_mesh_distance import point_mesh_edge_distance
from pytorch3d.ops import knn_gather
from pytorch3d.io import load_obj
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds

from network.networks import Generator
from network.networks_no_MI import Generator as Generator_no_MI
from network.networks_default import Generator as Generator_default
from option.train_option import get_train_options
from utils.pc_util import draw_point_cloud
from utils.xyz_util import save_xyz_file
from pathlib import Path
from utils.midpoint_interpolation import MI_prediction
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--non_uniform', default=False, help='random or non random input import for mesh file')
parser.add_argument('--resume', '-e', type=str, required=True, help='experiment name')
parser.add_argument('--resume2', type=str, default='PU1K_non_uniform', help='Comparative model for Arbitrary scale')
parser.add_argument('--off_file', type=str, required=False, help='off file')
parser.add_argument('--path', type=str, required=False, help='h5 file')
parser.add_argument('--gen_attention',type=str,default='self')
parser.add_argument('--dis_attention',type=str,default='self')
parser.add_argument('--feat_ext',type=str,default='self')
parser.add_argument('--alpha',type=int,default=8)
parser.add_argument('--scale',type=str,default='self')
args = parser.parse_args()

resume      = args.resume
resume2     = args.resume2
non_uniform = args.non_uniform
off_file    = args.off_file
pathfile    = args.path

def plot_save_pcd(pcd, file, exp_name):

    #print(pcd.shape)
    
    image_save_dir=os.path.join("../vis_result",exp_name)
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    file_name=file.split("/")[-1].split('.')[0]
    img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
                           diameter=4)
    img=(img*255).astype(np.uint8)
    image_save_path=os.path.join(image_save_dir,file_name+".png")
    cv2.imwrite(image_save_path,img)

def save_predict_pcd(pcd, file, exp_name):

    predict_save_dir=os.path.join("../predict_result",exp_name)
    if os.path.exists(predict_save_dir)==False:
        os.makedirs(predict_save_dir)

    file_name=file.split("/")[-1].split('.')[0]
    predict_save_path=os.path.join(predict_save_dir,file_name+".xyz")
    save_xyz_file(pcd, predict_save_path)
    
    
def read_off(off_file, gt_nb_pt, input_nb_pt):
    '''
    input_nb_pt : Number of point of the input        (eg : 1024).
    gt_nb_pt    : Number of point of the ground truth (eg: 4096).
    '''
    mesh = o3d.io.read_triangle_mesh(off_file)

    # Convert to point cloud
    input = mesh.sample_points_poisson_disk(number_of_points=input_nb_pt)
    gt    = mesh.sample_points_poisson_disk(number_of_points=gt_nb_pt)

    gt    = np.asarray(gt.points)
    gt    = torch.from_numpy(gt)
    gt    = gt[np.newaxis,...].float().cuda()#1,nb,3

    input = np.asarray(input.points)
    input = torch.from_numpy(input)
    input = input[np.newaxis,...].float().cuda()#1,nb,3

    return gt, input


def nonuniform(gt):

    nb_point  = gt.shape[1]
    nb_input  = int(nb_point/4)
    randindex = torch.randperm(nb_point)[:nb_input]
    input     = gt[:,randindex,:]

    return input
    
def get_cd_loss(pred, gt):

    '''
    pred and gt is (N, P1, D)
    '''
    #print('Pred :: ', pred.shape)
    #print('GT :: ',gt.shape)
    cham_x,_ = chamfer_distance(pred, gt,batch_reduction = "mean", point_reduction = "mean", norm = 2)
    #print('Shape cham_x :: ',cham_x.shape)

    return cham_x

def p2f_distance(mesh_path, pred):
    '''
    pred      : (N, P1, D)
    mesh_path : path
    '''
    pcl  = Pointclouds(points=pred)
    mesh = IO().load_mesh(mesh_path, device='cuda')
    p2f_value = point_mesh_edge_distance(mesh, pcl)
    
    return p2f_value

def hd_distance(pred, gt):
    '''
    pred and gt is (N, P1, D)
    '''

    pred = pred[0, ...]
    pred = pred.detach().cpu().numpy()
    gt = gt[0, ...]
    gt = gt.detach().cpu().numpy()
    hd_value1 = pcu.hausdorff_distance(pred, gt)
    
    return hd_value1

def hd_distance_manual(pred, gt):
    max_min12 = torch.tensor(-10)
    max_min21 = torch.tensor(-10)
    pred = pred[0,...]
    gt   = gt[0,...]
    
    for i in range(len(pred)):
        dist12    = torch.sqrt(torch.sum((gt-pred[i])**2, dim=1))
        mindist12 = torch.min(dist12)
        max_min12 = torch.max(max_min12, mindist12)

    for i in range(len(gt)):
        dist21    = torch.sqrt(torch.sum((pred-gt[i])**2, dim=1))
        mindist21 = torch.min(dist21)
        max_min21 = torch.max(max_min21, mindist21)    
    
    print('Hausdorff distance :: ',max_min12, max_min21)
    return torch.max(max_min12, max_min21)
    
def pc_normalization(pc):
    
    centroid = torch.mean(pc[..., :3], dim=1, keepdims=True)
    furthest_distance = torch.max(torch.norm(pc[...,:3]-centroid, dim=1))
    radius   = furthest_distance  # not very sure?
    #print('PC         :: \n',pc[..., :3].shape)
    #print('Centroid   :: \n',centroid.shape)
    #print('Difference :: \n', A.shape)
    #print('Radius     :: \n', radius)
    #print('Norm       :: \n',torch.norm(pc[...,:3]-centroid, dim=1))
    
    #normalization
    pc     = (pc-centroid)/radius
    
    return pc, centroid, radius

def patch_prediction(net, input_pc, patch_num_ratio=3,NUM_POINT=200):
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
    pc_seeds, seeds_idx  = sample_farthest_points(input_pc.contiguous(), K=seeds_num, random_start_point=False) 
    
    # Grouping
    _, idx, patches = knn_points(pc_seeds, input_pc, K = NUM_POINT, return_nn = True,return_sorted = True)
    patches         = patches.permute(0,2,1,3)
    
    #print('Input           :: ', input_pc.shape)
    #print('Number of Seeds  :: ', seeds_num)
    #print('Seed Point      :: ', pc_seeds.shape)
    #print('Seed Idx list   :: ', seeds_idx.shape)
    #print('Patches shape    :: ', patches.shape)

    input_list = []
    up_point_list = []
    cd_list    = [] 

    for k in range(seeds_num):
    #for k in tqdm(range(seeds_num)):
        patch    = patches[:, :, k, :]

        #Normalize
        patch, centroid, radius = pc_normalization(patch)
        patch    = patch.permute(0,2,1)
        
        #print('Patch    :: ',patch.shape)
        start    = time.time()
        up_point = net(patch)
        end      = time.time()
        
        centroid = centroid.permute(1,2,0)
        #print('UP point :: ', up_point.shape)
        #print('Centroid :: ', centroid.shape)
        up_point = up_point * radius + centroid

        #print('INPUT Patch :: ', patch.shape,' PRED Patch :: ', up_point.shape,' CD :: ',cd.item(),' Total time :: ',end - start)
    
        input_list.append(patch)       # 1,3,p
        up_point_list.append(up_point) # 1,3,4*p
    
    input_point = torch.cat(input_list, dim=-1)
    pred_pc     = torch.cat(up_point_list, dim=-1)
    
    return input_point, pred_pc, cd_list

def prediction(model, input, npoint, PATCH_NUM_RATIO, NUM_POINT):
    
    # Patch prediction
    _, pred_pc, _ = patch_prediction(model, input, patch_num_ratio=PATCH_NUM_RATIO, NUM_POINT=NUM_POINT)
    pred_pc       = torch.nan_to_num(pred_pc, nan=0.0)
    pred_pc       = pred_pc.permute(0,2,1)                    
    # FPS
    pred_pc_fps, _   = sample_farthest_points(pred_pc.contiguous(), K=npoint, random_start_point=False)
    
    return pred_pc_fps


def predict_from_h5():

    print(resume)
    print(non_uniform)
    print(off_file)
    print(pathfile)
    
    params                    = get_train_options()
    params['gen_attention']   = args.gen_attention
    params['dis_attention']   = args.dis_attention
    params['feat_ext']        = args.feat_ext
    params['scale']           = args.scale
    params['alpha']           = args.alpha
    exp_name                  = resume.split('/')[-2]
    exp_name2                 = resume2
    df_all                    = pd.DataFrame({})

    #for epoch in [49]:
    for epoch in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:

        # Check if the epochs exist
        ckpt_model = '../checkpoints/'+str(exp_name)+'/G_iter_'+str(epoch)+'.pth'
        print(ckpt_model)

        if Path(ckpt_model).is_file():
            
            # Load model
            if params['gen_attention']=='self' and params['dis_attention']=='self' and params['scale']=='self' and params['feat_ext']=='self':
            #Default PUGAN
                if pathfile.find('PUGAN')!=-1:
                    model   = Generator(params)                 
                else:
                    model   = Generator_default(params)
            
            elif params['scale']=='self' or params['feat_ext']=='P3DConv':
            #no scale, P3Dconv, mamba2
                model      = Generator_no_MI(params)

            elif params['gen_attention']=='mamba2' and params['dis_attention']=='mamba2':
            #Gen2 and Dis2
                model      = Generator_no_MI(params)
            
            else:
            #with scale, P3Dconv, mamba2
                # model with arbitrary scale
                model         = Generator(params)
                model_default = Generator_default(params)
        
                # Default model 
                
                if exp_name.find('non_uniform')!=-1 :
                    ckpt_model2 = '../checkpoints/'+str(exp_name2)+'/G_iter_'+str(epoch)+'.pth'                
                else:
                    exp_name2   = 'PU1K_uniform'
                    ckpt_model2 = '../checkpoints/'+str(exp_name2)+'/G_iter_'+str(epoch)+'.pth'
                #print('XXXX Default Model :: ',exp_name2)
                checkpoint2 = torch.load(ckpt_model2)
                model_default.load_state_dict(checkpoint2)
                model_default.eval().cuda()

            checkpoint = torch.load(ckpt_model)                
            model.load_state_dict(checkpoint)
            model.eval().cuda()

            cd_list         = []            
            cd_MI_list      = []
            cd_double_list  = []
            cd_both_list    = []

            hd_list         = []            
            hd_MI_list      = []
            hd_double_list  = []
            hd_both_list    = []

            p2f_list        = []            
            p2f_MI_list     = []
            p2f_double_list = []
            
            nb_input        = 2048
            NUM_POINT       = 256
            PATCH_NUM_RATIO = 3

            #Read file
            gt_h5_file = h5py.File(pathfile)
            mesh_h5    = gt_h5_file['mesh'][:]
            gt_h5      = gt_h5_file['gt'][:]
            input_h5   = gt_h5_file['input'][:]
            
            if params['scale']=='arbitrary':
                gt_h5      = gt_h5_file['gt'][:]
                input_h5   = gt_h5_file['gt'][:]
                list_alpha = [params['alpha']]
                
            input_h5 = torch.from_numpy(input_h5).float().cuda()
            gt_h5    = torch.from_numpy(gt_h5).float().cuda()

            #print(input_h5.shape, gt_h5.shape)
            #for path in range(1):
            for path in tqdm(range(len(input_h5))):
                input     = input_h5[path,:,:]
                gt        = gt_h5[path,:,:]
                input     = input[np.newaxis,...]
                gt        = gt[np.newaxis,...]
                mesh_file = '../MC_5k/'+str(mesh_h5[path].decode())
                
                ##########################
                
                if params['scale']=='arbitrary':
                
                    alpha       = random.choice(list_alpha)
                    data_npoint = gt.shape[1]
                    npoint      = gt.shape[1]
                    K           = int(npoint/alpha) #nb input point
                    if non_uniform==True:
                        # Non-uniform
                        sample_idx = utils.nonuniform_sampling(data_npoint, sample_num=K)
                        input      = input[:,sample_idx,:]
                    else:
                        input, _   = sample_farthest_points(input.contiguous(), K=K, random_start_point=False)
                    
                    input   = torch.nan_to_num(input, nan=0.0)
                    input, centroid, furthest_distance = pc_normalization(input)
                    input   = input.permute(0,2,1)
                    
                    # Patch prediction (Arbitrary Scale)
                    start         = time.time()
                    inputMI       = MI_prediction(input, alpha)
                    inputMI       = inputMI.permute(0,2,1)
                    pred_pc_fpsMI = prediction(model, inputMI, npoint, PATCH_NUM_RATIO, NUM_POINT)
                    end           = time.time()
                    pred_pc_fpsMI = (pred_pc_fpsMI * furthest_distance) + centroid
                    
                    # Patch prediction (Double Upsampling)
                    start         = time.time()
                    input         = input.permute(0,2,1)
                    input_fps     = prediction(model_default, input, 4*K, PATCH_NUM_RATIO, NUM_POINT)
                    pred_pc_fps   = prediction(model_default, input_fps, npoint, PATCH_NUM_RATIO, NUM_POINT)
                    end           = time.time()
                    pred_pc_fps   = (pred_pc_fps * furthest_distance) + centroid
                    input         = (input * furthest_distance) + centroid

                    #CD & HD & P2F
                    cd_MI          = get_cd_loss(pred_pc_fpsMI, gt)
                    cd_double      = get_cd_loss(pred_pc_fps, gt)
                    cd_both        = get_cd_loss(pred_pc_fps, pred_pc_fpsMI)
                    
                    hd_MI          = hd_distance(pred_pc_fpsMI, gt)
                    hd_double      = hd_distance(pred_pc_fps, gt)
                    hd_both        = hd_distance(pred_pc_fps, pred_pc_fpsMI)

                    p2f_MI         = p2f_distance(mesh_file, pred_pc_fpsMI)                    
                    p2f_double     = p2f_distance(mesh_file, pred_pc_fps)
                    
                    cd_MI_list.append(cd_MI.item())
                    cd_double_list.append(cd_double.item())
                    cd_both_list.append(cd_both.item())
                    
                    hd_MI_list.append(hd_MI)
                    hd_double_list.append(hd_double)
                    hd_both_list.append(hd_both)
                    
                    p2f_MI_list.append(p2f_MI.item())
                    p2f_double_list.append(p2f_double.item())

                    pred_pc_fpsMI = pred_pc_fpsMI[0, ...]
                    pred_pc_fpsMI = pred_pc_fpsMI.detach().cpu().numpy()

                    inputMI = inputMI[0, ...]
                    inputMI = inputMI.detach().cpu().numpy()                    
                    
                    #plot_save_pcd(pred_pc_fpsMI, pathfile.replace('.','_PRED_MI_'+str(epoch)+'.'), exp_name)#prediction
                    #plot_save_pcd(inputMI, pathfile.replace('.','_INPUT_MI_'+str(epoch)+'.'), exp_name)#prediction                    
                else:
                    input = torch.nan_to_num(input, nan=0.0)
                    input, centroid, furthest_distance = pc_normalization(input)
                    # Patch prediction
                    start = time.time()
                    _, pred_pc, _ = patch_prediction(model, input, patch_num_ratio=PATCH_NUM_RATIO, NUM_POINT=NUM_POINT)
                    end         = time.time()
    
                    pred_pc   = torch.nan_to_num(pred_pc, nan=0.0)
                    pred_pc   = pred_pc.permute(0,2,1)
    
                    # FPS
                    pred_pc_fps, _   = sample_farthest_points(pred_pc.contiguous(), K=4*nb_input, random_start_point=False)
                    pred_pc_fps   = (pred_pc_fps * furthest_distance) + centroid
                    input         = (input * furthest_distance) + centroid
                
                    ##########################
                    
                    #CD & HD & P2F
                    cd  = get_cd_loss(pred_pc_fps, gt)
                    hd  = hd_distance(pred_pc_fps, gt)
                    #hd2 = hd_distance_manual(pred_pc_fps, gt)
                    p2f_value = p2f_distance(mesh_file, pred_pc_fps)
                    cd_list.append(cd.item())
                    hd_list.append(hd)
                    p2f_list.append(p2f_value.item())
                    
                #Vizualisations
                pred_pc_fps = pred_pc_fps[0, ...]
                pred_pc_fps = pred_pc_fps.detach().cpu().numpy()

                input = input[0, ...]
                input = input.detach().cpu().numpy()        
                #plot_save_pcd(pred_pc_fps, pathfile.replace('.','_PRED_'+str(epoch)+'.'), exp_name)#prediction
                #plot_save_pcd(input, pathfile.replace('.','_INPUT_'+str(epoch)+'.'), exp_name)#prediction
                
                #print('File :: '+mesh_file.split('/')[-1]+' INPUT PC :: ', input.shape,' GT PC :: ', gt.shape,' PRED PC  :: ', pred_pc_fps.shape,' Centroid :: ',centroid.shape,' CD :: ',cd.item(),' HD :: ',hd,' p2f :: ',p2f_value.item(),' Total time :: ',end - start)
                #print('--iter :: ['+str(path)+str('/')+str(len(input_h5))+'] File :: '+mesh_file.split('/')[-1]+' CD :: ',cd.item(),' HD :: ',hd,' p2f :: ',p2f_value.item(),' Total time :: ',end - start)
                
                # Remove the implemented batch
                #data         = data[0, ...]
                #pred_pc_fps  = pred_pc_fps[0, ...]
            
            if params['scale']=='arbitrary':
                df_temp = pd.DataFrame({'model':[exp_name], 'epoch':[epoch], 'dataset_uniform':[non_uniform]
                    , 'CD_MI':[np.mean(cd_MI_list)], 'CD_double':[np.mean(cd_double_list)], 'CD_both':[np.mean(cd_both_list)]
                    , 'HD_MI':[np.mean(hd_MI_list)], 'HD_double':[np.mean(hd_double_list)], 'HD_both':[np.mean(hd_both_list)]
                    , 'P2F_MI':[np.mean(p2f_MI_list)], 'P2F_double':[np.mean(p2f_double_list)]
                    })
                df_all  = pd.concat([df_all,df_temp], axis=0)
                df_all.to_csv('Final_Result_'+exp_name+'_'+str(non_uniform)+'.csv', index=None, sep=',')
                msg2 ="{}, {}, {}, {}, {}, {}".format(
                            exp_name, epoch, non_uniform, np.mean(cd_both_list), np.mean(hd_both_list), (np.mean(p2f_MI_list)+np.mean(p2f_double_list))/2
                        )
                print(msg2)

            else:
                df_temp = pd.DataFrame({'model':[exp_name], 'epoch':[epoch], 'dataset_uniform':[non_uniform], 'CD':[np.mean(cd_list)], 'HD':[np.mean(hd_list)], 'P2F':[np.mean(p2f_list)]})
                df_all  = pd.concat([df_all,df_temp], axis=0)
                df_all.to_csv('Final_Result_'+exp_name+'_'+str(non_uniform)+'.csv', index=None, sep=',')
        
                msg2 ="{}, {}, {}, {}, {}, {}".format(
                            exp_name, epoch, non_uniform, np.mean(cd_list), np.mean(hd_list), np.mean(p2f_list)
                        )
                print(msg2)

    
    #print('MEAN CD  :: ',np.mean(cd_list))
    #print('MEAN HD  :: ',np.mean(hd_list))
    #print('MEAN P2F :: ',np.mean(p2f_list))
    
if __name__ == "__main__":
    
    #batch   = 10
    #nbpoint = 5000
    #dim     = 3
    
    #pc=torch.rand(batch, nbpoint,dim).cuda()
    #patch_prediction(pc, patch_num_ratio=4)
    #predict_from_mesh()
    predict_from_h5()
