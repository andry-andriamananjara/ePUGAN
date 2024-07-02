import torch
import torch.nn as nn
import os,sys
sys.path.append('../')
#from auction_match import auction_match
#import pointnet2.pointnet2_utils as pn2_utils
import math
#from knn_cuda import KNN
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.utils import masked_gather
from pytorch3d.loss.chamfer import chamfer_distance

class Loss(nn.Module):
    def __init__(self,radius=1.0):
        super(Loss,self).__init__()
        self.radius=radius
        #self.knn_uniform=KNN(k=2,transpose_mode=True)
        #self.knn_repulsion=KNN(k=20,transpose_mode=True)
        # self.knn_uniform=knn_points(K=2,transpose_mode=True)
        # self.knn_repulsion=knn_points(K=20,transpose_mode=True)

    def get_cd_loss(self,pred,gt,radius=1.0):
        '''
        pred and gt is (N, P1, D)
        '''
        #print('Pred :: ', pred.shape)
        #print('GT :: ',gt.shape)
        cham_x,_ = chamfer_distance(pred, gt,batch_reduction = "mean", point_reduction = "mean", norm = 2)
        #print('Shape cham_x :: ',cham_x.shape)

        return cham_x

    def get_uniform_loss(self,pcd,percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
        B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
        npoint=int(N*0.05)
        loss=0

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
            # idx=pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous()) #b N nsample            
            expect_len=math.sqrt(disk_area)

            # grouped_pcd=pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)#B C N nsample
            # grouped_pcd=tp.grouping_operation(pcd.contiguous(), idx)  # (B, 3, npoint, nsample)

            # print('Grouped pcd before :: ',grouped_pcd.shape)
            # grouped_pcd=grouped_pcd.permute(0,3,1,2) #B N nsample C
            grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C
            # print('Grouped pcd after :: ',grouped_pcd.shape)

            #(N, P1, K),_,_
            dist,_,_ = knn_points( grouped_pcd,grouped_pcd,K = 2, return_nn = True,return_sorted = True)
            # dist,_=self.knn_uniform(grouped_pcd,grouped_pcd)

            uniform_dist=dist[:,:,1:] #B*N nsample 1

            uniform_dist=torch.abs(uniform_dist+1e-8)
            uniform_dist=torch.mean(uniform_dist,dim=1)
            uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
            mean_loss=torch.mean(uniform_dist)
            mean_loss=mean_loss*math.pow(p*100,2)
            loss+=mean_loss
        return loss/len(percentage)
        
    def get_repulsion_loss(self,pcd,h=0.0005):
        # dist,idx=self.knn_repulsion(pcd,pcd)#B N k
        # self.knn_repulsion=knn_points(K=20,transpose_mode=True)
        pcd = torch.transpose(pcd, 1, 2) #batch_size,n_points,n_dims
        dist,idx,_ = knn_points( pcd,pcd,K = 20, return_nn = True,return_sorted = True)

        '''
        pcd: is batch_size,n_dims,n_points => batch_size,n_points,n_dims (pytorch3d) 
        dist: N, P1, K (pytorch3d)
        '''

        dist=dist[:,:,1:5]**2 #top 4 cloest neighbors
        loss=torch.clamp(-dist+h,min=0)
        loss=torch.mean(loss)
        #print(loss)
        return loss
    
    def get_discriminator_loss(self,pred_fake,pred_real):
        real_loss=torch.mean((pred_real-1)**2)
        fake_loss=torch.mean(pred_fake**2)
        loss=real_loss+fake_loss
        return loss
    def get_generator_loss(self,pred_fake):
        fake_loss=torch.mean((pred_fake-1)**2)
        return fake_loss
    def get_discriminator_loss_single(self,pred,label=True):
        if label==True:
            loss=torch.mean((pred-1)**2)
            return loss
        else:
            loss=torch.mean((pred)**2)
            return loss

if __name__=="__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Torch CUDA available :: ',device)
    loss=Loss().cuda()
    point_cloud=torch.rand(4,4096,3).cuda()

    #loss=Loss()
    #point_cloud=torch.rand(4,4096,3)

    uniform_loss=loss.get_uniform_loss(point_cloud)
    repulsion_loss=loss.get_repulsion_loss(point_cloud)

