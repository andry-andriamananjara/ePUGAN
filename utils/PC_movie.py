import os
import glob
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def img_to_video(image_files,video_name):

    fps = 20
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    
    exp_name = 'video'
    path_to_videos=os.path.join("../vis_result",exp_name)
    if os.path.exists(path_to_videos)==False:
        os.makedirs(path_to_videos)
    
    clip.write_videofile(path_to_videos+video_name)
    
if __name__ == "__main__":
    
    image_files = glob.glob('/scratch/project_2009906/PUGAN-Pytorch/MC_5k/Mydataset/PU_LiDAR/images/*.jpeg')
    video_name  = 'PC_original.mp4'
    img_to_video(image_files,video_name)
    
    #image_files = glob.glob('/scratch/project_2009906/PUGAN-Pytorch/vis_result/trainpu1kGen2_non_uniform_xyz/*.jpeg')
    #video_name  = 'CroppedVideo.mp4'
    #img_to_video(image_files, video_name)
    
    #image_files = glob.glob('/scratch/project_2009906/PUGAN-Pytorch/vis_result/trainpu1kGen2_non_uniform_xyz/PRED*.png')
    #video_name = 'Upsampled.mp4'
    #img_to_video(image_files, video_name)
    
    
#cd /scratch/project_2009906/PUGAN-Pytorch/utils/