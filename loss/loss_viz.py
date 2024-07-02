import os, sys
sys.path.append("../")
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_file(filename,gan=2):
    file    = open(filename,'r')
    content = file.readlines()
    
    if gan==2:
        if filename.find('21550454')!=-1 or filename.find('21560160')!=-1:
            step  = 27
            G_ = content[step::4]
            D_ = content[step+1::4]
    
        elif filename.find('21597022')!=-1 or filename.find('21597028')!=-1 or filename.find('21594286'):
            step  = 24
            G_ = content[step::3]
            D_ = content[step+1::3]
    
        else:
            step  = 22
            G_ = content[step::3]
            D_ = content[step+1::3]
        
        G_text = []
        D_text = []
        
        for i in tqdm(range(len(G_))):
        #for i in tqdm(range(20)):
            G_text.append(G_[i].split(','))
            D_text.append(D_[i].split(','))
        file.close()
        
        df_D    = pd.DataFrame(D_text)
        df_D[5] = df_D[3]
        df_D[3] = df_D[3].apply(lambda x: 'D_loss')
        df_D[5] = df_D[5].apply(lambda x: 1000 if x is None else float(x.split(':')[1].strip()))
        df_D[6] = df_D[4].apply(lambda x: 1 if x is None else float(x.split(':')[1].strip()))
        
        df_G    = pd.DataFrame(G_text)
        df_G[5] = df_G[3]
        df_G[3] = df_G[3].apply(lambda x: 'G_loss')
        df_G[5] = df_G[5].apply(lambda x: 1000 if x is None else float(x.split(':')[1].strip()))
        df_G[6] = df_G[4].apply(lambda x: 1 if x is None else float(x.split(':')[1].strip()))    
    
    else:
        if filename.find('21509751')!=-1:
            step  = 26
            G_ = content[step::2]
        else:
            step  = 25
            G_ = content[step::]
        
        G_text = []
        
        for i in tqdm(range(len(G_))):
        #for i in tqdm(range(20)):
            G_text.append(G_[i].split(','))
        file.close()
        
        df_D    = pd.DataFrame()

        df_G    = pd.DataFrame(G_text)
        df_G[5] = df_G[3]
        df_G[3] = df_G[3].apply(lambda x: 'G_loss')
        df_G[5] = df_G[5].apply(lambda x: 1000 if x is None else float(x.split(':')[1].strip()))
        df_G[6] = df_G[4].apply(lambda x: 1 if x is None else float(x.split(':')[1].strip()))    
        
    
    return df_G, df_D

def double_visual(df_G, df_D, filename):
    
    image_save_dir='Loss_Viz'
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)
    
    # Loss G & Loss D
    plt.clf()
    plt.figure()
    plt.plot([i for i in range(len(df_G))], df_G[5], 'g', label='Loss_G')
    plt.plot([i for i in range(len(df_D))], df_D[5], 'r', label='Loss_D')
    plt.title('Generator vs Discriminator Loss value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.yscale('log', base=10)
    plt.legend()
    
    image_save_path=os.path.join(image_save_dir, filename.split('/')[-1][:-4]+"_Loss_G_Loss_D.png")
    plt.savefig(image_save_path)

    # Iteration time Loss G & Loss D
    plt.clf()
    plt.figure()
    plt.plot([i for i in range(len(df_G))], df_G[6], 'g', label='Loss_G')
    plt.plot([i for i in range(len(df_D))], df_D[6], 'r', label='Loss_D')
    plt.title('Generator vs Discriminator iteration time')
    plt.xlabel('Iteration')
    plt.ylabel('Time')
    plt.yscale('log', base=10)
    plt.legend()
    
    image_save_path=os.path.join(image_save_dir, filename.split('/')[-1][:-4]+"_G_D_iteration_time.png")
    plt.savefig(image_save_path)

def single_visual(df_G,filename):
    
    image_save_dir='Loss_Viz'
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)
    
    # Loss G & Loss D
    plt.clf()
    plt.figure()
    plt.plot([i for i in range(len(df_G))], df_G[5], 'g', label='Loss_G')
    plt.title('Generator Loss value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.yscale('log', base=10)
    plt.legend()
    
    image_save_path=os.path.join(image_save_dir, filename.split('/')[-1][:-4]+"_Loss_G.png")
    plt.savefig(image_save_path)

    # Iteration time Loss G & Loss D
    plt.clf()
    plt.figure()
    plt.plot([i for i in range(len(df_G))], df_G[6], 'g', label='Loss_G')
    plt.title('Generator iteration time')
    plt.xlabel('Iteration')
    plt.ylabel('Time')
    plt.yscale('log', base=10)
    plt.legend()
    
    image_save_path=os.path.join(image_save_dir, filename.split('/')[-1][:-4]+"_G_iteration_time.png")
    plt.savefig(image_save_path)


def main():
    #filename   = '../SLURM/slurm-21597022.out'  #PU1K uniform
    #filename   = '../SLURM/slurm-21597028.out'  #PU1K non uniform
    #filename   = '../SLURM/slurm-21550454.out'  #PU1K uniform (GenDis Mamba)     Expt
    #filename   = '../SLURM/slurm-21560160.out'  #PU1K non uniform (GenDis Mamba) Expt
    #filename   = '../SLURM/slurm-21594287.out'  #PU1K uniform (GenDisP3Dconv)
    #filename   = '../SLURM/slurm-21594288.out'  #PU1K non uniform (GenDisP3Dconv)
    #filename   = '../SLURM/slurm-21594286.out'  #PU1K non uniform (DisP3Dconv)    
    #filename   = '../SLURM/slurm-21594284.out'  #PU1K uniform (DisP3Dconv)        
    #filename   = '../SLURM/slurm-21594280.out'  #PU1K uniform (GenP3Dconv)   
    #filename   = '../SLURM/slurm-21594281.out'  #PU1K non uniform (GenP3Dconv)   

    #filename   = '../SLURM/slurm-21630431.out'  #PU1K uniform (P3Dconv 40 Epoch)   
    #filename   = '../SLURM/slurm-21630440.out'  #PU1K non uniform (P3Dconv 40 Epoch) 
    #filename   = '../SLURM/slurm-21631102.out'  #PU1K uniform (Gen2 40 Epoch)
    #filename   = '../SLURM/slurm-21631115.out'  #PU1K non uniform (Gen2 40 Epoch)
    #filename   = '../SLURM/slurm-21631127.out'  #PU1K uniform (Gen2P3Dconv 40 Epoch)
    #filename   = '../SLURM/slurm-21631130.out'  #PU1K non uniform (Gen2P3Dconv 40 Epoch)
    
    # Around 99 Epochs & Failed
    #filename   = '../SLURM/slurm-21635006.out'  #PU1K uniform (P3Dconv)   
    #filename   = '../SLURM/slurm-21635004.out'  #PU1K non uniform (P3Dconv) 98Epochs
    #filename   = '../SLURM/slurm-21635225.out'  #PU1K uniform (Gen2)
    #filename   = '../SLURM/slurm-21635227.out'  #PU1K non uniform (Gen2)
    #filename   = '../SLURM/slurm-21635119.out'  #PU1K uniform (Gen2P3Dconv)
    #filename   = '../SLURM/slurm-21635120.out'  #PU1K non uniform (Gen2P3Dconv)

    # (Re-run) Around 33 Epochs
    #filename   = '../SLURM/slurm-21675390.out'  #PU1K uniform (P3Dconv)   
    #filename   = '../SLURM/slurm-21675407.out'  #PU1K non uniform (P3Dconv) 98Epochs
    #filename   = '../SLURM/slurm-21675472.out'  #PU1K uniform (Gen2P3Dconv)
    #filename   = '../SLURM/slurm-21675475.out'  #PU1K non uniform (Gen2P3Dconv)

    # Arbitrary Scale ()
    filenlist   = [
        #'../SLURM/slurm-21772771.out',  #PU1K uniform (Arbscale)   
        '../SLURM/slurm-21774253.out',  #PU1K non uniform (Arbscale)
        '../SLURM/slurm-21774256.out',  #PU1K uniform (Gen2Arbscale)
        '../SLURM/slurm-21774274.out'  #PU1K non uniform (Gen2Arbscale)
    ]
    
    for filename in filenlist:
        df_G, df_D = load_file(filename)
        print(df_G.head(5))
        print(df_D.head(5))
        double_visual(df_G, df_D, filename)
    
    #filename   = '../SLURM/slurm-21505141.out'  #PU1K non uniform (Dis)   
    #filename   = '../SLURM/slurm-21505138.out'  #PU1K uniform (Dis)
    #filename   = '../SLURM/slurm-21509751.out'  #PU1K uniform (Gen)
    #filename   = '../SLURM/slurm-21505134.out'  #PU1K non uniform (Gen)
    
    #df_G, _ = load_file(filename,gan=1)
    #print(df_G.head(5))
    #single_visual(df_G,filename)
    
if __name__ == "__main__":
    main()
