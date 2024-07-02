import os, sys
import argparse
sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', type=str, required=True)
parser.add_argument('--isTrain', type=str, required=True)
parser.add_argument('--arbitrary_scale', action='store_true', help='arbitrary scale')
args   = parser.parse_args()

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.sample_farthest_points import sample_farthest_points_naive

from tqdm import tqdm
import open3d as o3d
import glob 
import numpy as np
import h5py
import torch
import math

from utils.xyz_util import save_xyz_file


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

def train_test_index(file_name, niter):
    
    path_ = os.path.join("/scratch/project_2009916/PUGAN-Pytorch/data", file_name)
    file = open(path_,'w')
    for item in range(niter):
    	file.write(str(item)+"\n")
    file.close()



def off_to_patches(folder, off_file, uniform=True, seeds_num=200, seeds_size=256, arbitrary_scale=1, up_ratio=4):
    '''
    input_nb_pt : 1024
    gt_nb_pt    : 4096
    '''
    up_rate = int(arbitrary_scale*up_ratio) #default up_rate = 4
    
    # Read OFF file
    mesh      = o3d.io.read_triangle_mesh(off_file)
    general   = mesh.sample_points_poisson_disk(number_of_points=100000)
    general   = np.asarray(general.points)
    general   = torch.from_numpy(general)
    general   = general[np.newaxis,...].float().cuda()#1,nb,3

    pc_seeds, seeds_idx  = sample_farthest_points(general.contiguous(), K=seeds_num, random_start_point=False) 
    _, idx, gt_patches = knn_points(pc_seeds, general, K = up_rate*seeds_size, return_nn = True,return_sorted = True)
    gt_patches    = gt_patches.permute(0,2,1,3)
    
    # Save as XYZ file
    arb_scale_list  = []
    input_list      = []
    gt_list         = []
    mesh_file       = []
    for k in range(seeds_num):
        gt_pt          = gt_patches[:, :, k, :]
        mesh_file      += [off_file]
        arb_scale_list += [arbitrary_scale]
        # uniform input
        if uniform:
            input_pt, _  = sample_farthest_points(gt_pt.contiguous(), K=seeds_size, random_start_point=False)
            #print(gt_pt.shape, input_pt.shape)
            input_pt = np.asarray(input_pt[0, ...].cpu())

            gt_pt    = np.asarray(gt_pt[0, ...].cpu())
            gt_list.append(gt_pt)
            input_list.append(input_pt)
        
        # non uniform input
        else:
            gt_pt    = np.asarray(gt_pt[0, ...].cpu())
            random_index = np.random.choice(up_rate*seeds_size, seeds_size, replace=False)
            input_pt     = gt_pt[random_index,:]
            #print(gt_pt.shape, input_pt.shape)
            gt_list.append(gt_pt)
            input_list.append(input_pt)
            
        #save_xyz_file(gt_pt, path_gt.replace('.xyz','_'+str(k)+'.xyz'))
        #save_xyz_file(input_pt, path_input.replace('.xyz','_'+str(k)+'.xyz'))
    
    return input_list, gt_list, mesh_file, arb_scale_list



def off_patch_whole(folder, off_file, uniform=True, nb_input=1024, nb_gt=4096, arbitrary_scale=1, up_ratio=4):
    '''
    input_nb_pt : 1024
    gt_nb_pt    : 4096
    '''
    mesh      = o3d.io.read_triangle_mesh(off_file)
    general   = mesh.sample_points_poisson_disk(number_of_points=nb_gt)
    general   = np.asarray(general.points)

    input     = mesh.sample_points_poisson_disk(number_of_points=nb_input)
    input     = np.asarray(input.points) #uniform
    
    if not uniform:
        random_index = np.random.choice(nb_gt, nb_input, replace=False)
        input_pt     = general[random_index]
    
    return input, general, arbitrary_scale

#######################################################
#######################################################
#######################################################

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
    
    for mesh in ['Mesh']:
        
        #for folder in ['train','test']:
        for folder in [isTrain]:

            seeds_size   = 256 #PUGAN
            seeds_num    = 200 #PUGAN
            
            #seeds_size   = 1024 #PU1K
            #seeds_num    = 100  #PU1K
            if folder=='test':
                seeds_num  = 50
                seeds_size = 2048
            
            offlist  = glob.glob(path_off+'/*.off')
            print('Processing folder : ',folder)
            
            
            scale_list = [1]
            if arb_scale:
                seeds_size = 512
                #scale_list = [4.5, 2, 8, 9.5]
                scale_list = [8]
                
            for alpha in scale_list:
                
                p_arb_list   = []
                mesh_train   = []
                mesh_test    = []
                gt_uni       = []
                input_uni    = []
                gt_nonuni    = []
                input_nonuni = []
                
                arb_whole          = []
                gt_uni_whole       = []
                input_uni_whole    = []
                gt_nonuni_whole    = []
                input_nonuni_whole = []        


                print('alpha : ',alpha)
                for i in tqdm(range(len(offlist))):
                #for i in range(2):
                    
                    input, gt, mesh_file, arb = off_to_patches(folder, offlist[i], uniform=True, seeds_num=seeds_num, arbitrary_scale=alpha, seeds_size=seeds_size)
                    gt_uni     += gt
                    input_uni  += input
                    mesh_train += mesh_file
                    p_arb_list += arb
                    
                    input, gt, mesh_file, arb = off_to_patches(folder, offlist[i], uniform=False, seeds_num=seeds_num, arbitrary_scale=alpha, seeds_size=seeds_size)
                    gt_nonuni    += gt
                    input_nonuni += input
                    
                    if folder=='test':
                        input_unif, general_unif, arb = off_patch_whole(folder, offlist[i], uniform=True, nb_input=seeds_size, arbitrary_scale=alpha, nb_gt=int(4*alpha)*seeds_size)
                        gt_uni_whole    += [general_unif]
                        input_uni_whole += [input_unif]
                        mesh_test       += [offlist[i]]
                        arb_whole       += [arb]
                        
                        input_non_unif, general_non_unif, arb = off_patch_whole(folder, offlist[i], uniform=False, nb_input=seeds_size, arbitrary_scale=alpha, nb_gt=int(4*alpha)*seeds_size)
                        gt_nonuni_whole    += [general_non_unif]
                        input_nonuni_whole += [input_non_unif]
                
                # Patches
                folder_name=folder
                if arb_scale:
                    folder_name=folder_name+'_arbitrary_scale_'+str(alpha)
                
                distribution = 'uniform_'+str(seeds_size)+'_'+str(4*seeds_size)+'_'
                file_name    = path_off+'/'+distribution+folder_name+'.h5'
                to_h5file(file_name, input_uni, gt_uni, mesh_train, p_arb_list)
                
                distribution = 'non_uniform_'+str(seeds_size)+'_'+str(4*seeds_size)+'_'
                file_name    = path_off+'/'+distribution+folder_name+'.h5'
                to_h5file(file_name, input_nonuni, gt_nonuni, mesh_train, p_arb_list)
                
                # Patches from whole off
                if folder=='test':
                    distribution = 'uniform_'+str(seeds_size)+'_'+str(4*seeds_size)+'_'
                    file_name    = path_off+'/'+distribution+folder_name+'_whole.h5'
                    to_h5file(file_name, input_uni_whole, gt_uni_whole, mesh_test, arb_whole)
                    
                    distribution = 'non_uniform_'+str(seeds_size)+'_'+str(4*seeds_size)+'_'
                    file_name    = path_off+'/'+distribution+folder_name+'_whole.h5'
                    to_h5file(file_name, input_nonuni_whole, gt_nonuni_whole, mesh_test, arb_whole)
