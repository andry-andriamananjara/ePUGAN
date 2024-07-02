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
from tqdm import tqdm

from network.networks import Generator
from option.train_option import get_train_options
from utils.pc_util import draw_point_cloud
from utils.xyz_util import save_xyz_file
import point_cloud_utils as pcu

parser = argparse.ArgumentParser()
parser.add_argument('--non_uniform', default=False, help='random or non random input import for mesh file')
parser.add_argument('--resume', '-e', type=str, required=True, help='experiment name')
parser.add_argument('--off_file', type=str, required=False, help='off file')
parser.add_argument('--path', type=str, required=False, help='h5 file')
parser.add_argument('--gen_attention',type=str,default='self')
parser.add_argument('--dis_attention',type=str,default='self')
parser.add_argument('--feat_ext',type=str,default='self')
args = parser.parse_args()

resume      = args.resume
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
    
    #print('PC         :: \n',pc[..., :3])
    #print('Centroid   :: \n',centroid)
    #print('Difference :: \n', pc[...,:3]-centroid)
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

        #cd = get_cd_loss(up_point, gt_patch)        
        #cd_list.append(cd.item())
        #print('INPUT Patch :: ', patch.shape,' PRED Patch :: ', up_point.shape,' CD :: ',cd.item(),' Total time :: ',end - start)
    
        input_list.append(patch)       # 1,3,p
        up_point_list.append(up_point) # 1,3,4*p
    
    input_point = torch.cat(input_list, dim=-1)
    pred_pc     = torch.cat(up_point_list, dim=-1)
    
    return input_point, pred_pc, cd_list

def predict_from_mesh():

    print(resume)
    print(non_uniform)
    
    # Load model
    model      = Generator()
    exp_name   = resume.split('/')[-2]
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint)
    model.eval().cuda()
    
    path_list  = glob.glob(off_file+"/*.off")
    
    cd_list         = []
    nb_input        = 4096
    NUM_POINT       = 256
    PATCH_NUM_RATIO = 3
    dynamic         = True

    if non_uniform:
        exp_name = os.path.join(exp_name,"mesh/non_uniform")
    else:
        exp_name = os.path.join(exp_name,"mesh/uniform")
    
    for path in path_list:
    #for path in [path_list[0], path_list[1], path_list[2], path_list[3],]:
        
        #Read off file
        gt, input = read_off(off_file=path, gt_nb_pt=4*nb_input, input_nb_pt=nb_input)
        if non_uniform:
            input    = nonuniform(gt)

        input, centroid, furthest_distance = pc_normalization(input)
        
        # Patch prediction
        start = time.time()
        input, pred_pc, _ = patch_prediction(model, input, patch_num_ratio=PATCH_NUM_RATIO, NUM_POINT=NUM_POINT)
        end         = time.time()
        
        input     = input.permute(0,1,2)
        pred_pc   = pred_pc.permute(0,2,1)
        pred_pc   = (pred_pc * furthest_distance) + centroid
        input     = (input * furthest_distance) + centroid
        
        # FPS
        pred_pc_fps, _   = sample_farthest_points(pred_pc.contiguous(), K=4*nb_input, random_start_point=False)
        
        #Chamfer distance
        cd = get_cd_loss(pred_pc_fps, gt)        
        cd_list.append(cd.item())
        
        print('INPUT PC :: ', input.shape,' GT PC :: ', gt.shape,' PRED PC  :: ', pred_pc_fps.shape,' Centroid :: ',centroid.shape,' CD :: ',cd.item(),' Total time :: ',end - start)
        
        #Vizualisations
        pred_pc_fps = pred_pc_fps[0, ...]
        pred_pc_fps = pred_pc_fps.detach().cpu().numpy()
        save_predict_pcd(pred_pc_fps, path, exp_name)
        #plot_save_pcd(pred_pc_fps, path.replace('.','_PRED_'+str(4*nb_input)+'x'+str(NUM_POINT)+'x'+str(PATCH_NUM_RATIO)+'.'), exp_name)#prediction
        #plot_save_pcd(input, path.replace('.','_INPUT_'+str(nb_input)+'x'+str(NUM_POINT)+'x'+str(PATCH_NUM_RATIO)+'.'), exp_name)#prediction
        
        # Remove the implemented batch
        #data         = data[0, ...]
        #pred_pc_fps  = pred_pc_fps[0, ...]
        
    print('MEAN CD :: ',np.mean(cd_list))


def predict_from_h5():

    print(resume)
    print(non_uniform)
    print(off_file)
    print(pathfile)
    
    params                    = get_train_options()
    params['gen_attention']   = args.gen_attention
    params['dis_attention']   = args.dis_attention
    params['feat_ext']        = args.feat_ext
    
    # Load model
    exp_name   = resume.split('/')[-2]
    checkpoint = torch.load(resume)
    model      = Generator(params)
    model.load_state_dict(checkpoint)
    model.eval().cuda()
    
    cd_list         = []
    hd_list         = []
    p2f_list        = []
    nb_input        = 2048
    NUM_POINT       = 256
    PATCH_NUM_RATIO = 3

    #Read file
    gt_h5_file = h5py.File(pathfile)
    gt_h5     = gt_h5_file['gt'][:]
    input_h5  = gt_h5_file['input'][:]
    mesh_h5   = gt_h5_file['mesh'][:]
    
    input_h5 = torch.from_numpy(input_h5).float().cuda()
    gt_h5    = torch.from_numpy(gt_h5).float().cuda()

    #print(input_h5.shape, gt_h5.shape)
    #for path in range(5):
    for path in tqdm(range(len(input_h5))):
    #for path in range(len(input_h5)):
        input     = input_h5[path,:,:]
        gt        = gt_h5[path,:,:]
        input     = input[np.newaxis,...]
        gt        = gt[np.newaxis,...]
        mesh_file = '../MC_5k/'+str(mesh_h5[path].decode())
        
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
        #save_predict_pcd(pred_pc_fps, path, exp_name)
        #plot_save_pcd(pred_pc_fps, pathfile.replace('.','_PRED_'+str(4*nb_input)+'x'+str(NUM_POINT)+'x'+str(PATCH_NUM_RATIO)+'.'), exp_name)#prediction
        #plot_save_pcd(input, pathfile.replace('.','_INPUT_'+str(nb_input)+'x'+str(NUM_POINT)+'x'+str(PATCH_NUM_RATIO)+'.'), exp_name)#prediction
        
        #print('File :: '+mesh_file.split('/')[-1]+' INPUT PC :: ', input.shape,' GT PC :: ', gt.shape,' PRED PC  :: ', pred_pc_fps.shape,' Centroid :: ',centroid.shape,' CD :: ',cd.item(),' HD :: ',hd,' p2f :: ',p2f_value.item(),' Total time :: ',end - start)
        #print('--iter :: ['+str(path)+str('/')+str(len(input_h5))+'] File :: '+mesh_file.split('/')[-1]+' CD :: ',cd.item(),' HD :: ',hd,' p2f :: ',p2f_value.item(),' Total time :: ',end - start)
        
        # Remove the implemented batch
        #data         = data[0, ...]
        #pred_pc_fps  = pred_pc_fps[0, ...]
        
    print('MEAN CD  :: ',np.mean(cd_list))
    print('MEAN HD  :: ',np.mean(hd_list))
    print('MEAN P2F :: ',np.mean(p2f_list))
    
if __name__ == "__main__":
    
    #batch   = 10
    #nbpoint = 5000
    #dim     = 3
    
    #pc=torch.rand(batch, nbpoint,dim).cuda()
    #patch_prediction(pc, patch_num_ratio=4)
    #predict_from_mesh()
    predict_from_h5()
