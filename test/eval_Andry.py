import argparse
import os, sys
sys.path.append("../")

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--isUniform", action='store_true', default=False)
parser.add_argument('--resume', type=str, required=True)
parser.add_argument('--datatest', type=str, required=True)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data.data_loader import PUNET_Dataset, PUNET_Dataset_uniform
#from chamfer_distance import chamfer_distance
#from auction_match import auction_match
#import pointnet2.utils.pointnet2_utils as pn2_utils
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
from hausdorff import hausdorff_distance
import point_cloud_utils as pcu
import math

def get_cd_loss(pred, gt, pcd_radius):
#def get_emd_loss(pred, gt, pcd_radius):
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

def get_uniform(pcd, percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
#def get_uniform(pcd, percentage=[0.004],radius=1.0):
    B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
    npoint=int(N*0.05)
    loss=0
    
    #print('Point cloud Uniform shape :: ',pcd.shape)
    
    '''
    pcd (B,N,C)       ==> points (N Batchs, P points, D dimension) (pytorch3d)
    further_point_idx ==> (N Batchs, D dimension) or (N Batchs, P points, D dimension) (pytorch3d)
    new_xyz (B,C,N)   ==> (N Batchs, P points, D dimension) (pytorch3d)
    '''

    # further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
    _, further_point_idx = sample_farthest_points(pcd.contiguous(), K=npoint, random_start_point=False)        

    # new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
    new_xyz = masked_gather(points=pcd.contiguous(), idx=further_point_idx) #N,P,D

    for p in percentage:
        nsample=int(N*p)
        r=math.sqrt(p*radius)
        disk_area=math.pi*(radius**2)/N

        #_,(N, P1, K),(N, P1, K, D)
        _, idx, grouped_pcd=ball_query(p1=pcd.contiguous(),p2=new_xyz.contiguous(),
                                    lengths1 = None,lengths2 = None, K = nsample,radius = r,return_nn = True,)
        expect_len=math.sqrt(disk_area)

        # print('Grouped pcd before :: ',grouped_pcd.shape)
        grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C
        # print('Grouped pcd after :: ',grouped_pcd.shape)

        #(N, P1, K),_,_
        dist,_,_ = knn_points( grouped_pcd,grouped_pcd,K = 2, return_nn = True,return_sorted = True)

        uniform_dist=dist[:,:,1:] #B*N nsample 1

        uniform_dist=torch.abs(uniform_dist+1e-8)
        uniform_dist=torch.mean(uniform_dist,dim=1)
        uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
        mean_loss=torch.mean(uniform_dist)
        mean_loss=mean_loss*math.pow(p*100,2)
        loss+=mean_loss

    return loss/len(percentage)

if __name__ == '__main__':
    param=get_train_options()
    model = Generator()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    eval_dst = PUNET_Dataset_uniform(gt_h5_file_path=args.datatest, uniform=args.isUniform, isTrain=False)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=args.workers)

    emd_list = []
    cd_list  = []
    hd_list  = []
    uniform_list = []
    with torch.no_grad():
        for itr, batch in enumerate(eval_loader):

            points, gt, radius = batch
            points = points[..., :3].permute(0,2,1).float().cuda().contiguous()
            gt = gt[..., :3].float().cuda().contiguous()
            radius = radius.float().cuda()
            preds  = model(points)  # points.shape[1])
            preds  = preds.permute(0,2,1).contiguous()
            
            #print(preds.shape, gt.shape)
            #emd = get_emd_loss(preds, gt, radius)
            
            a = torch.cat(torch.unbind(preds,dim=1),dim=0).data.cpu().numpy() #preds.data.cpu().numpy()
            b = torch.cat(torch.unbind(gt,dim=1),dim=0).data.cpu().numpy() #gt.data.cpu().numpy()
            #hd = pcu.hausdorff_distance(a, b)
            hd = hausdorff_distance(a,b, distance="euclidean")
            cd = get_cd_loss(preds, gt, radius)

            uniform_ = get_uniform(preds)

            #print(' -- iter {}, emd {}, cd {}.'.format(itr, emd, cd))
            print(' -- iter {}, input {}, pred {}, gt {}, cd {}, hd {}, uniform {}.'.format(itr, points.shape, preds.shape, gt.shape, cd, hd, uniform_))
            #emd_list.append(emd.item())
            cd_list.append(cd.item())
            hd_list.append(hd)
            uniform_list.append(uniform_.item())
            
    #print('mean emd     : {}'.format(np.mean(emd_list)))
    print('mean cd       : {}'.format(np.mean(cd_list)))
    print('mean hd       : {}'.format(np.mean(hd_list)))
    print('mean uniform  : {}'.format(np.mean(uniform_list)))
