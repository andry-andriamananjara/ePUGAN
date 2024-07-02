import glob
import pandas as pd
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
#parser.add_argument('--resume', type=str, required=True)
parser.add_argument('--resume', type=str, required=False)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data.data_loader import PUNET_Dataset
#from chamfer_distance import chamfer_distance
#from auction_match import auction_match
#import pointnet2.utils.pointnet2_utils as pn2_utils
import importlib
from network.networks import Generator
from option.train_option import get_train_options

from pytorch3d.ops.utils import masked_gather
from pytorch3d.loss.chamfer import chamfer_distance
import point_cloud_utils as pcu

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



#def get_cd_loss(pred, gt, pcd_radius):
#    cost_for, cost_bac = chamfer_distance(gt, pred)
#    cost = 0.5 * cost_for + 0.5 * cost_bac
#    cost /= pcd_radius
#    cost = torch.mean(cost)

#    return cost


if __name__ == '__main__':

    folders = glob.glob('../checkpoints/Fext1-4/', recursive = True)
    #folders = glob.glob('../checkpoints/train20240321mamba2-7*/', recursive = True)
    #folders = glob.glob('../checkpoints/train20240321/', recursive = True)
    #folders = os.walk(directory)
    df = pd.read_csv('Final_Result.csv', sep=';')
    #df = pd.DataFrame({})
    
    param=get_train_options()
    eval_dst = PUNET_Dataset(h5_file_path='../Patches_noHole_and_collected.h5', split_dir=param['test_split'], isTrain=False)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=args.workers)
    
    for folder in folders:
        
        mean_cd       = []
        mean_hd       = []
        chkpt_number  = []
        chkpt_name    = []
        
        counter = 9
        files = glob.glob(folder+'G*.pth')
        
        for file in files:
            print(file)
            model = Generator()
            checkpoint = torch.load(file)        
            #checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            model.eval().cuda()

            emd_list = []
            cd_list  = []
            hd_list  = []
            
            with torch.no_grad():
                for itr, batch in enumerate(eval_loader):
                    points, gt, radius = batch
                    points = points[..., :3].permute(0,2,1).float().cuda().contiguous()
                    gt = gt[..., :3].float().cuda().contiguous()
                    radius = radius.float().cuda()
                    preds = model(points)  # points.shape[1])
                    preds=preds.permute(0,2,1).contiguous()
                    
                    #print(preds.shape, gt.shape)
                    #emd = get_emd_loss(preds, gt, radius)
                    
                    
                    a = torch.cat(torch.unbind(preds,dim=1),dim=0).data.cpu().numpy() #preds.data.cpu().numpy()
                    b = torch.cat(torch.unbind(gt,dim=1),dim=0).data.cpu().numpy() #gt.data.cpu().numpy()
                    hd = pcu.hausdorff_distance(a, b)
                    cd = get_cd_loss(preds, gt, radius)
                    #print(' -- iter {}, emd {}, cd {}.'.format(itr, emd, cd))
                    #print(' -- iter {}, cd {}, hd {}.'.format(itr, cd, hd))
                    #emd_list.append(emd.item())
                    cd_list.append(cd.item())
                    hd_list.append(hd)
            
            #print('mean emd: {}'.format(np.mean(emd_list)))
            print('mean cd: {}'.format(np.mean(cd_list)))
            print('mean hd: {}'.format(np.mean(hd_list)))
            
            mean_cd.append(np.mean(cd_list))
            mean_hd.append(np.mean(hd_list))
            chkpt_number.append(counter)
            chkpt_name.append(file.split('/')[-1][6:9])
            counter = counter+10
        
        train_name = [folder.split('//')[-1] for i in range(len(chkpt_name))]
        df_temp = pd.DataFrame({'train':train_name,'checkpoint':chkpt_number,'checkpoint_list':chkpt_name,'CD':mean_cd, 'HD':mean_hd})
        
        df = pd.concat([df, df_temp], axis=0)
    df.to_csv('Final_Result.csv', sep=';', index=None)
    print(df)
    