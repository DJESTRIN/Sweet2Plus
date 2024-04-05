""" RunDeepCAD by David James Estrin
A set of functions that bridge current two photon pipeline with DeepCAD API. 
(1) All subjects data will be moiton corrected with suite2p. 
(2) Motion corrected tif stacks will be copied to common path (--training_data_path), which will be the training dataset.
(3) Following training, individual tif stacks for each mouse will be brough throuch DeepCAD's Neural net and processed for denoising. 
"""
from deepcad.train_collection import training_class
from deepcad.movie_display import display, display_img
from deepcad.utils import get_first_filename,download_demo
import matplotlib.pyplot as plt
import argparse

def DeepCAD(datasets_path, pth_dir, n_epochs = 5, GPU = '0', 
            train_datasets_size = 500, patch_xy = 150, patch_t = 150,
            overlap_factor = 0.25, num_workers = 0, visualize_images_per_epoch = True,
            save_test_images_per_epoch = True, display_images = True):

    """ Show video prior to training """
    if display_images:
        display_filename = get_first_filename(datasets_path)
        print('\033[1;31mDisplaying the first raw file -----> \033[0m')
        print(display_filename)
        display_length = 300  # the frames number of the noise movie
        # normalize the image and display
        display(display_filename, display_length=display_length, norm_min_percent=1, norm_max_percent=98)

    train_dict = {
        # dataset dependent parameters
        'patch_x': patch_xy,
        'patch_y': patch_xy,
        'patch_t': patch_t,
        'overlap_factor':overlap_factor,
        'scale_factor': 1,                  # the factor for image intensity scaling
        'select_img_num': 100000,           # select the number of images used for training (use all frames by default)
        'train_datasets_size': train_datasets_size,
        'datasets_path': datasets_path,
        'pth_dir': pth_dir,
        # network related parameters
        'n_epochs': n_epochs,
        'lr': 0.00005,                       # initial learning rate
        'b1': 0.5,                           # Adam: bata1
        'b2': 0.999,                         # Adam: bata2
        'fmap': 16,                          # the number of feature maps
        'GPU': GPU,
        'num_workers': num_workers,
        'visualize_images_per_epoch': visualize_images_per_epoch,
        'save_test_images_per_epoch': save_test_images_per_epoch}
    
    tc = training_class(train_dict)
    tc.run()

    if display_images:
        display_filename = tc.result_display
        print('\033[1;31mDisplaying denoised file of the last epoch-----> \033[0m')
        print(display_filename)
        # normalize the image and display
        img = display_img(display_filename,norm_min_percent=1, norm_max_percent=99)
        plt.imshow(img,cmap=plt.cm.gray,vmin=0,vmax=255)
        plt.axis('off')
        plt.show()
    
    return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, required=True, help='The full path to current animals registered data')
    parser.add_argument('--training_data_path',type=str, required=True, help='The full path to a common folder where all registered data will be dropped for all animals')
    parser.add_argument('--deepcad_output_path',type=str, required=True, help='The full path to current animals DeepCAD output data')
    datasets_pathoh = r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058\suite2p\plane0\reg_tif'  # folder containing tif files for training
    pth_diroh = r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058\DeepCADoutput'   
    DeepCAD(datasets_pathoh,pth_diroh)

