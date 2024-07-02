import os, sys
sys.path.append("../")
import glob
import pandas as pd
import matplotlib.pyplot as plt


def viz_epoch(filename):

    image_save_dir='Epoch_Viz'
    if os.path.exists(image_save_dir)==False:
        os.makedirs(image_save_dir)

    df = pd.read_csv(filename, sep=',')
    
    # CD
    plt.clf()
    plt.figure()
    plt.plot(df['epoch'], df['CD'], 'magenta', label='CD')
    plt.plot(df['epoch'], df['HD'], 'lime', label='HD')
    plt.plot(df['epoch'], df['P2F'], 'tab:orange', label='P2F')
    plt.title('CD vs HD vs P2F')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.yscale('log', base=10)
    plt.legend()
    
    image_save_path=os.path.join(image_save_dir, filename.split('/')[-1].replace('.csv',"_.png"))
    plt.savefig(image_save_path)

if __name__ == "__main__":
    
    #list_file = glob.glob('*Result_PU1K_non_uniform_test*.csv')+glob.glob('*Result_PU1K_uniform_test*.csv')
    list_file = glob.glob('*trainpu1kGen2_P3Dconv_non_uniform_test*.csv')+glob.glob('*trainpu1kGen2_P3Dconv_uniform_test*.csv')
    #list_file = glob.glob('*trainpu1kP3Dconv_non_uniform*.csv')+glob.glob('*trainpu1kP3Dconv_uniform*.csv')
    for filename in list_file:
        viz_epoch(filename)
    