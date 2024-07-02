import os,sys
sys.path.append("../")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, required=True, help='experiment name')
parser.add_argument('--debug', action='store_true', help='specify debug mode')
parser.add_argument('--use_gan',action='store_true')
parser.add_argument('--batch_size',type=int,default=12)
parser.add_argument('--alpha',type=int,default=8)
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--gen_attention',type=str,default='self')
parser.add_argument('--dis_attention',type=str,default='self')
parser.add_argument('--feat_ext',type=str,default='self')
parser.add_argument('--dataname',type=str,default='punet')
parser.add_argument('--scale',type=str,default='self')
parser.add_argument('--uniform',action='store_true', default=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
sys.path.append('../')
import torch
from network.networks import Generator,Discriminator
from data.data_loader import PUNET_Dataset, PUNET_PU1K_Dataset

import time
from option.train_option import get_train_options
from utils.Logger import Logger
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from loss.loss import Loss
import datetime
import torch.nn as nn

from pytorch3d.ops import sample_farthest_points
from utils.midpoint_interpolation import MI_prediction
import utils.data_util as utils
import random

def xavier_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(args):
    start_t=time.time()
    print('Training input ::: ', args.feat_ext)
    params                    = get_train_options()
    params["exp_name"]        = args.exp_name
    params["patch_num_point"] = 1024
    params["batch_size"]      = args.batch_size
    params['use_gan']         = args.use_gan
    params['gen_attention']   = args.gen_attention
    params['dis_attention']   = args.dis_attention
    params['feat_ext']        = args.feat_ext
    params['scale']           = args.scale
    params['alpha']           = args.alpha
    params['uniform']         = args.uniform
    list_alpha                = [2, 4, 6, params['alpha']]
    if args.debug:
        params["nepoch"]=2
        params["model_save_interval"]=3
        params['model_vis_interval']=3

    log_dir=os.path.join(params["model_save_dir"],args.exp_name)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    tb_logger=Logger(log_dir)
    
    #for uniform
    #trainloader = PUNET_PU1K_Dataset(uniform=True, dataname = "punet")
    trainloader = PUNET_PU1K_Dataset(uniform=args.uniform, dataname = args.dataname)
    
    #for non uniform
    #trainloader = PUNET_Dataset(h5_file_path=params["dataset_dir"],split_dir=params['train_split'])
    #print(params["dataset_dir"])

    num_workers=4
    train_data_loader=data.DataLoader(dataset=trainloader,batch_size=params["batch_size"],shuffle=True,
                                      num_workers=num_workers,pin_memory=True,drop_last=True)
    

    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    G_model=Generator(params)
    #G_model.apply(xavier_init)
    G_model=torch.nn.DataParallel(G_model).to(device)
    D_model=Discriminator(params,in_channels=3)
    #D_model.apply(xavier_init)
    D_model=torch.nn.DataParallel(D_model).to(device)

    G_model.train()
    D_model.train()

    optimizer_D=Adam(D_model.parameters(),lr=params["lr_D"],betas=(0.9,0.999))
    optimizer_G=Adam(G_model.parameters(),lr=params["lr_G"],betas=(0.9,0.999))

    D_scheduler = MultiStepLR(optimizer_D,[50,80],gamma=0.2)
    G_scheduler = MultiStepLR(optimizer_G,[50,80],gamma=0.2)

    Loss_fn=Loss()

    print("preparation time is %fs" % (time.time() - start_t))
    iter=0
    for e in range(params["nepoch"]):
        D_scheduler.step()
        G_scheduler.step()
        for batch_id,(input_data, gt_data, radius_data) in enumerate(train_data_loader):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            input_data=input_data[:,:,0:3].permute(0,2,1).float().cuda()
            gt_data=gt_data[:,:,0:3].permute(0,2,1).float().cuda()

            start_t_batch=time.time()
            
            # For arbitrary scale
            if params['scale']=='arbitrary':
                
                alpha       = random.choice(list_alpha)
                data_npoint = gt_data.shape[2]
                npoint      = gt_data.shape[2]
                #print('******** uniform ',args.uniform)
                #print(input_data.shape, gt_data.shape)

                if not args.uniform:
                    # Non-uniform
                    sample_idx = utils.nonuniform_sampling(data_npoint, sample_num=int(npoint/alpha))
                    input_data = input_data[:,:,sample_idx]
                else:
                    input_data=input_data.permute(0,2,1)
                    input_data, _   = sample_farthest_points(input_data.contiguous(), K=int(npoint/alpha), random_start_point=False)
                    input_data=input_data.permute(0,2,1)
                    
                input_data     = MI_prediction(input_data, alpha)
                #print('xx input : ',input_data.shape)
            
            # Direct computation
            output_point_cloud=G_model(input_data)
            
            repulsion_loss = Loss_fn.get_repulsion_loss(output_point_cloud.permute(0, 2, 1))
            uniform_loss = Loss_fn.get_uniform_loss(output_point_cloud.permute(0, 2, 1))
            #print(output_point_cloud.shape,gt_data.shape)
            #emd_loss = Loss_fn.get_emd_loss(output_point_cloud.permute(0, 2, 1), gt_data.permute(0, 2, 1))
            emd_loss = Loss_fn.get_cd_loss(output_point_cloud.permute(0, 2, 1), gt_data.permute(0, 2, 1))

            if params['use_gan']==True:
                fake_pred = D_model(output_point_cloud.detach())
                d_loss_fake = Loss_fn.get_discriminator_loss_single(fake_pred,label=False)
                d_loss_fake.backward()
                optimizer_D.step()

                real_pred = D_model(gt_data.detach())
                d_loss_real = Loss_fn.get_discriminator_loss_single(real_pred, label=True)
                d_loss_real.backward()
                optimizer_D.step()

                d_loss=d_loss_real+d_loss_fake

                fake_pred=D_model(output_point_cloud)
                g_loss=Loss_fn.get_generator_loss(fake_pred)

                #print(repulsion_loss.item(), uniform_loss.item(), emd_loss.item(), g_loss.item())
                total_G_loss=params['uniform_w']*uniform_loss+params['emd_w']*emd_loss+ \
                repulsion_loss*params['repulsion_w']+ g_loss*params['gan_w']
            else:
                #total_G_loss = params['uniform_w'] * uniform_loss + params['emd_w'] * emd_loss + \
                #               repulsion_loss * params['repulsion_w']
                total_G_loss=params['emd_w'] * emd_loss + \
                               repulsion_loss * params['repulsion_w']

            #total_G_loss=emd_loss
            total_G_loss.backward()
            optimizer_G.step()

            current_lr_D=optimizer_D.state_dict()['param_groups'][0]['lr']
            current_lr_G=optimizer_G.state_dict()['param_groups'][0]['lr']

            tb_logger.scalar_summary('repulsion_loss', repulsion_loss.item(), iter)
            tb_logger.scalar_summary('uniform_loss', uniform_loss.item(), iter)
            tb_logger.scalar_summary('emd_loss', emd_loss.item(), iter)
            if params['use_gan']==True:
                tb_logger.scalar_summary('d_loss', d_loss.item(), iter)
                tb_logger.scalar_summary('g_loss', g_loss.item(), iter)
            tb_logger.scalar_summary('lr_D', current_lr_D, iter)
            tb_logger.scalar_summary('lr_G', current_lr_G, iter)
            
            msg0='-------------------------------------------------------------------------------------------'
            msg1="{:0>8},{}:{}, [{}/{}], {}: {},{}:{}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                "total_G_loss",
                total_G_loss.item(),
                "iter time",
                (time.time() - start_t_batch)
            )
            msg2="{:0>8},{}:{}, [{}/{}], {}: {},{}:{}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                "total_D_loss",
                d_loss.item(),
                "iter time",
                (time.time() - start_t_batch)
            )
            print(msg0)
            print(msg1)
            print(msg2)
            
            iter+=1
        if (e+1) % params['model_save_interval'] == 0 and e > 0:
            model_save_dir = os.path.join(params['model_save_dir'], params['exp_name'])
            if os.path.exists(model_save_dir) == False:
                os.makedirs(model_save_dir)
            D_ckpt_model_filename = "D_iter_%d.pth" % (e)
            G_ckpt_model_filename = "G_iter_%d.pth" % (e)
            D_model_save_path = os.path.join(model_save_dir, D_ckpt_model_filename)
            G_model_save_path = os.path.join(model_save_dir, G_ckpt_model_filename)
            torch.save(D_model.module.state_dict(), D_model_save_path)
            torch.save(G_model.module.state_dict(), G_model_save_path)


if __name__=="__main__":
    import colored_traceback
    train(args)
    
    