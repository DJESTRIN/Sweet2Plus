import numpy as np
import ipdb
import glob, os
import tifffile
import tqdm
np.seterr(all='raise')

def correct_streaking(input_image_path,output_image_path,delta_threshold):
    # Find all input images
    images = glob.glob(os.path.join(input_image_path,'*.tif*'))

    # Create output image path
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)

    # Loop over all images
    for imdir in images:
        image=tifffile.imread(imdir)

        # Determine whether image has streaking based on threshold of first dirivative
        average_column = image.sum(axis=1)
        spikes = np.where(average_column>delta_threshold)
        
        # Run de-streaking
        immin = image.min()
        immax = image.max()
        if np.any(spikes):
            for i,column in enumerate(image.T):
                if (i+100)>len(image.T):
                    segment=segment
                else:
                    finish=i+100
                    segment = image.T[i:finish]
                medsig = segment.mean(axis=0)
                ans=column-medsig
                ans[ans<immin]=immin
                ans[ans>immax]=immax
                image.T[i]=ans
        
        # Save image to new folder
        #image=(image-image.min()/(image.max()-image.min()))*255
        original_filename = os.path.basename(imdir)
        outputname=os.path.join(output_image_path,original_filename.replace('Ch2','Ch1'))
        tifffile.imwrite(outputname,image)

def run_whole_folder(path_in,path_out):
    all_dirs = glob.glob(os.path.join(path_in,'*24*'))
    for diroh in tqdm.tqdm(all_dirs):
        foldername=os.path.basename(diroh)
        outputfolder=os.path.join(path_out,foldername)
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        correct_streaking(diroh,outputfolder,20)   


if __name__=='__main__':
    path = r'D:\2p_drn_inhibition\twophotonrecordings\24-2-23'
    output_path = r'D:\2p_drn_inhibition\twophotonrecordings\24-2-23_destriped'
    run_whole_folder(path,output_path)