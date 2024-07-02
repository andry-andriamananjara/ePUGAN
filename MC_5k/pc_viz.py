import os, sys
sys.path.append("../")

import argparse
import utils.data_util as utils
import h5py
import cv2
import numpy as np
import open3d as o3d
import glob
from utils.pc_util import draw_point_cloud
from Common import point_operation
from data.data_loader import PUNET_Dataset, PUNET_Dataset_uniform

parser = argparse.ArgumentParser()
parser.add_argument('--file_type', type=str, required=True, help='punet or pu1k or xyz or pu1k_scale')
parser.add_argument('--path', type=str, required=True, help='path file')
parser.add_argument('--uniform',action='store_true', default=False)
parser.add_argument('--xyz',action='store_true', default=False)
parser.add_argument('--mesh',action='store_true', default=False)
args = parser.parse_args()

def plot_save_pcd(pcd, file, exp_name):

    image_save_dir=os.path.join("viz_sample",exp_name)

    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    file_name=file.split("/")[-1].split('.')[0]
    #img = draw_point_cloud(pcd, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi,
    #                       diameter=4)

    img = draw_point_cloud(pcd, zrot=0, xrot=0, yrot=0 / 180.0 * np.pi,
                           diameter=4)

    img=(img*255).astype(np.uint8)
    image_save_path=os.path.join(image_save_dir,file_name+".png")
    cv2.imwrite(image_save_path,img)


def plot_mesh(file, exp_name):

    mesh      = o3d.io.read_triangle_mesh(file)
    #general   = mesh.sample_points_poisson_disk(number_of_points=100000)
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])
    
    image_save_dir=os.path.join("viz_sample",exp_name)
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)
    
    file_name=file.split("/")[-1].split('.')[0]
    image_save_path=os.path.join(image_save_dir,file_name+"_MESH.png")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(image_save_path)
    vis.destroy_window()
    
        
def plot_h5():
    
    file_type = args.file_type
    path      = args.path
    uniform   = args.uniform
    exp_name  = "viz_test"

    if file_type == 'punet':
        gt_h5_file = h5py.File(path)
        if uniform:
            #path = "Mydataset/PUNET/uniform_256_1024_test.h5"
            #path = "Mydataset/PUNET/non_uniform_256_1024_train.h5"
            
            print(gt_h5_file.keys())
            gt     = gt_h5_file['gt'][:]
            input  = gt_h5_file['input'][:]
        else:
            print(gt_h5_file.keys())
            gt     = gt_h5_file['gt'][:]
            input  = gt_h5_file['input'][:]
    
    elif file_type == 'pugan':
        #/scratch/project_2009906/PUGAN-Pytorch/MC_5k/Mydataset/PUGAN
        gt_h5_file = h5py.File(path)
        gt    = gt_h5_file['gt'][:]
        input = gt_h5_file['input'][:]
        #gt    = gt_h5_file['poisson_1024'][:]
        #input = gt_h5_file['poisson_256'][:]
        
    elif file_type == 'pu1k':
        gt_h5_file = h5py.File(path)
        if uniform:
            #path = "Mydataset/PUNET/uniform_256_1024_test.h5"
            #path = "Mydataset/PUNET/non_uniform_256_1024_train.h5"
            print(gt_h5_file.keys())
            gt     = gt_h5_file['poisson_1024'][:]
            input  = gt_h5_file['poisson_256'][:]
        else:
            print(gt_h5_file.keys())
            gt     = gt_h5_file['poisson_1024'][:]
            input  = gt_h5_file['poisson_1024'][:]
    
    elif file_type == 'pu1k_scale':
        dataset=PUNET_Dataset_uniform(uniform=False, dataname = "pu1k_scale")
        (input_data,gt_data,radius_data, scale)=dataset.__getitem__(0)
        print(
            'Input   :: ', input_data[0].shape, input_data[1].shape, input_data[2].shape,
            'GT      :: ', gt_data[0].shape, gt_data[1].shape, gt_data[2].shape,
            'Radius  :: ', radius_data[0].shape, radius_data[1].shape, radius_data[2].shape,
            'Scale  :: ', scale[0].shape, scale[1].shape, scale[2].shape
        )
    
    else:
        pcd = np.loadtxt(path)
        plot_save_pcd(pcd, path, exp_name)
    
    
    
    
    ## Plot
    if file_type == 'pu1k_scale':
        for num_idx in range(3):
            plot_save_pcd(input_data[num_idx], path+'_'+str(num_idx)+'_scale_'+str(scale[num_idx])+'_gt.', exp_name)
            plot_save_pcd(gt_data[num_idx], path+'_'+str(num_idx)+'_scale_'+str(scale[num_idx])+'_input.', exp_name)
    
    else:
        for num_idx in range(10):
            print(gt.shape, input.shape)
            pcd     = gt[num_idx,:,:]
            plot_save_pcd(pcd, path.replace('.',str(num_idx)+'_gt.'), exp_name)
    
            pcd     = input[num_idx,:,:]
            if not uniform:
                idx = point_operation.nonuniform_sampling(1024, sample_num=256)
                pcd = gt[num_idx,:,:]
                pcd = pcd[idx,:]
            plot_save_pcd(pcd, path.replace('.',str(num_idx)+'_input.'), exp_name)    

if __name__ == "__main__":
    
    if args.xyz:
        exp_name  = "viz_test"
        path      = args.path
        pcd       = o3d.io.read_point_cloud(path)
        npcd      = np.asarray(pcd.points)
        plot_save_pcd(npcd, path, exp_name)
        
    elif args.mesh:
        exp_name  = "viz_test"
        path      = args.path
        path_list = glob.glob(path+'/*.off')
        
        for i in range(10):
            plot_mesh(path_list[i], exp_name)
        
    else:
        plot_h5()
        
        
        
        