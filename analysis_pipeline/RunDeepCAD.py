""" RunDeepCAD by David James Estrin
A set of functions that bridge current two photon pipeline with DeepCAD API. 
(1) All subjects data will be moiton corrected with suite2p. 
(2) Motion corrected tif stacks will be copied to common path (--training_data_path), which will be the training dataset.
(3) Following training, individual tif stacks for each mouse will be brough throuch DeepCAD's Neural net and processed for denoising. 
"""
from deepcad.train_collection import training_class
from deepcad.test_collection import testing_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename
import matplotlib.pyplot as plt
import argparse

def DeepCAD(datasets_path, pth_dir, n_epochs = 10, GPU = '0', 
            train_datasets_size = 10000, patch_xy = 150, patch_t = 150,
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

    return

def DeepCADtest(data_path, output_path,model_path, GPU = '0', 
            test_datasize = 10000, patch_xy = 150, patch_t = 150,
            overlap_factor = 0.25, num_workers = 0, visualize_images_per_epoch = True,
            save_test_images_per_epoch = True, display_images = True):
    
    test_dict = {
        # dataset dependent parameters
        'patch_x': patch_xy,
        'patch_y': patch_xy,
        'patch_t': patch_t,
        'overlap_factor':overlap_factor,
        'scale_factor': 1,                  # the factor for image intensity scaling
        'test_datasize': test_datasize,     # the number of frames to be tested
        'datasets_path': data_path,     # folder containing all files to be tested
        'pth_dir': model_path,                 # pth file root path
        'denoise_model' : model_path,    # A folder containing all models to be tested
        'output_dir' : output_path,         # result file root path

        # network related parameters
        'fmap': 16,                          # the number of feature maps
        'GPU': GPU,
        'num_workers': num_workers,
        'visualize_images_per_epoch': visualize_images_per_epoch,
        'save_test_images_per_epoch': save_test_images_per_epoch}
    
    tc = testing_class(test_dict)
    tc.run()
   

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, required=True, help='The full path to current animals registered data')
    parser.add_argument('--output_data_path',type=str, required=True, help='The full path to a common folder where all registered data will be dropped for all animals')
    parser.add_argument('--model_path',type=str, required=True, help='The full path to current animals DeepCAD output data')
    args = parser.parse_args()

    #datasets_pathoh = r'C:\2p_drn_inhitbition_deepcad\input'  # folder containing tif files for training
    #pth_diroh = r'C:\2p_drn_inhitbition_deepcad\output'  
    #DeepCAD(datasets_pathoh,pth_diroh)
    #DeepCADtest(r'C:\2p_drn_inhitbition_deepcad\input',r'C:\2p_drn_inhitbition_deepcad\output\processed_data',r'C:\2p_drn_inhitbition_deepcad\output\models')
    
    DeepCADtest(args.data_path,args.output_data_path,args.model_path)

