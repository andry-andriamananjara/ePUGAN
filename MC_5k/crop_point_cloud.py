import os,sys
import argparse
sys.path.append('../')
parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', type=str, required=True)
parser.add_argument('--isTrain', type=str, required=True)
parser.add_argument('--arbitrary_scale', action='store_true', help='arbitrary scale')
args   = parser.parse_args()

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


def to_h5file(file_name, input, gt, mesh_file, arb_ratio):
    #print('xxgt    :: ', len(gt))
    #print('xxinput :: ', len(input))
    #print('xxmesh  :: ', len(mesh_file))
    #print('xxArb   :: ', len(arb_ratio))

    #print('xxgt    :: ', gt)
    #print('xxinput :: ', input)
    
    h5f = h5py.File(file_name,'w')
    
    h5f.create_dataset('gt', data              = gt)
    h5f.create_dataset('input', data           = input)
    h5f.create_dataset('mesh', data            = mesh_file)
    h5f.create_dataset('arbitrary_scale', data = arb_ratio)
    print(h5f.keys())
    print(h5f['gt'], h5f['input'], h5f['mesh'], h5f['arbitrary_scale'])
    h5f.close

if __name__=="__main__":
    
    datasetdir = args.datasetdir
    isTrain    = args.isTrain
    isTrain    = isTrain.lower()
    arb_scale  = args.arbitrary_scale
    print('Dataset dir    : ',datasetdir)
    print('Train/Test     : ', isTrain)
    print('Arbirary Scale : ', arb_scale)
    
    #path_off = 'Mydataset/PUNET/' 
    path_off = datasetdir

    pcd_save_dir=os.path.join(path_off,"xyz_file")

    if os.path.exists(pcd_save_dir)==False:
        os.makedirs(pcd_save_dir)

    
    #for folder in ['train','test']:
    for folder in [isTrain]:
        seeds_size   = 1024
        seeds_num    = 100
        if folder=='test':
            seeds_num  = 1
            seeds_size = 10000
        
        offlist  = glob.glob(path_off+'/*.pcd')+glob.glob(path_off+'/*.xyz')
        print('Processing folder : ',folder)

        p_arb_list   = []
        mesh_train   = []
        mesh_test    = []
        gt_uni       = []
        input_uni    = []
        gt_nonuni    = []
        input_nonuni = []

        scale_list   = [1]
        alpha        = 1
        print('alpha : ',alpha)

        for i in tqdm(range(len(offlist))):
        #for i in range(2):
            pcd     = o3d.io.read_point_cloud(offlist[i])
            downpcd = pcd.voxel_down_sample(voxel_size=0.8)
            filename   = offlist[i].split('/')[-1]
            filename   = os.path.join(pcd_save_dir,filename.replace('.pcd','.xyz').replace('.xyz','_cropped.xyz'))
            o3d.io.write_point_cloud(filename, downpcd)
            