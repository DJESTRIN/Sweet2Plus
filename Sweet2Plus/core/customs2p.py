""" Custom S2P by David James Estrin
"""
import suite2p as s2p
import matplotlib.pyplot as plt 
import os,glob
import seaborn as sns
import ipdb
import numpy as np
import cv2
import tqdm
from PIL import Image
sns.set_style('whitegrid')

class get_s2p():
    """ get suite 2P: This class is meant to run suite2P without the gui. """
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        #Set input and output directories
        self.datapath=datapath
        self.resultpath=os.path.join(self.datapath,'figures/')
        self.resultpath_so=os.path.join(self.datapath,'figures/serialoutput/')
        self.resultpath_neur=os.path.join(self.datapath,'figures/neuronal/')
        self.include_mask=True
        self.cellthreshold=cellthreshold

        if not os.path.exists(self.resultpath): #Make the figure directory
            os.mkdir(self.resultpath)
        if not os.path.exists(self.resultpath_so): #Make subfolder for serialoutput/behavioral data
            os.mkdir(self.resultpath_so)
        if not os.path.exists(self.resultpath_neur): #Make subfolder for neural data
            os.mkdir(self.resultpath_neur)


        #Set suite2P ops
        self.ops = s2p.default_ops()
        self.ops['batch_size'] = batch_size # we will decrease the batch_size in case low RAM on computer
        self.ops['threshold_scaling'] = threshold_scaling # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
        self.ops['fs'] = fs # sampling rate of recording, determines binning for cell detection
        self.ops['tau'] = tau # timescale of gcamp to use for deconvolution
        #self.ops['input_format']="bruker"
        self.ops['blocksize']=blocksize
        self.ops['reg_tif']=reg_tif
        self.ops['reg_tif_chan2']=reg_tif_chan2
        self.ops['denoise']=denoise

        #Set up datapath
        self.db = {'data_path': [self.datapath],}
    
    def __call__(self):
        self.animal_information()
        searchstring=os.path.join(self.datapath,'**/F_mlp.npy')
        res = glob.glob(searchstring,recursive=True)
        if not res:
            raise ValueError('We are finished with running suite2p')
            self.auto_run()
            self.get_reference_image()
        self.convert_motion_corrected_images()

    def animal_information(self):
        _,_,_,_,_,_,self.cage,self.mouse,_ = self.datapath.split('_')
        if 'Day1' in self.datapath:
            self.day=1
        if 'Day7' in self.datapath:
            self.day=7
        if 'Day14' in self.datapath:
            self.day=14

    def auto_run(self):
        self.output_all=s2p.run_s2p(ops=self.ops,db=self.db)

    def get_reference_image(self):
        filename=os.path.basename(self.datapath)
        filename_ref=os.path.join(self.resultpath_neur,f'{filename}referenceimage.jpg')
        filename_rigids=os.path.join(self.resultpath_neur,f'{filename}rigid.jpg')
        plt.figure(figsize=(20,20))
        plt.subplot(1, 4, 1)
        plt.imshow(self.output_all['refImg'],cmap='gray')

        plt.subplot(1, 4, 2)
        plt.imshow(self.output_all['max_proj'], cmap='gray')
        plt.title("Registered Image, Max Projection");

        plt.subplot(1, 4, 3)
        plt.imshow(self.output_all['meanImg'], cmap='gray')
        plt.title("Mean registered image")

        plt.subplot(1, 4, 4)
        plt.imshow(self.output_all['meanImgE'], cmap='gray')
        plt.title("High-pass filtered Mean registered image")
        plt.savefig(filename_ref)

    def stack_sort(self,path):
        path,_=path.split('_chan')
        _,path=path.split('file')
        path=int(path)
        return path

    def convert_motion_corrected_images(self):
        searchstring=os.path.join(self.datapath,'**/reg_tif/*.tif')
        drop_path = os.path.join(self.datapath,'motion_corrected_tif_seq/')

        if not os.path.exists(drop_path):
            os.mkdir(drop_path)
            convert=True
        else:
            convert=False

        if convert:
            tifstacks = glob.glob(searchstring,recursive=True)
            tifstacks.sort(key=self.stack_sort)
            slicecount=0
            stacks=[]
            
            #Collect all data to numpy array
            for image in tifstacks:
                dataset = Image.open(image)
                stack=[]
                for i in range(dataset.n_frames):
                    dataset.seek(i)
                    stack.append(np.array(dataset))
                stack=np.asarray(stack)
                stacks.append(stack)

            # Merge all stacks
            for i,stack in enumerate(stacks):
                if i==0:
                    Stack = stack
                else:
                    Stack = np.concatenate((Stack,stack),axis=0)

            Stack = ((Stack-Stack.min())/(Stack.max()-Stack.min()))*255
            Stack = Stack.astype(np.uint64)

            #Loop through frames and save to image
            for frame in Stack:
                filenameoh = os.path.join(drop_path,f'slice{slicecount}.tif')
                cv2.imwrite(filenameoh, frame)
                slicecount+=1
        
        corrected_imagesearch = os.path.join(drop_path,'*slice*.tif*')
        self.corrected_images = glob.glob(corrected_imagesearch)

        def sort_images(x):
            _,x=x.split('slice')
            x,_=x.split('.ti')
            return int(x)
        
        self.corrected_images.sort(key=sort_images)

class manual_classification(get_s2p):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold)
        return 
    
    def get_s2p_outputs(self):
        #Find planes and get recording/probability files
        search_path = os.path.join(self.datapath,'suite2p/plane*/')
        self.recording_files=[]
        self.probability_files=[]
        self.stat_files=[]
        planes = [result for result in glob.glob(search_path)]
        self.recording_files.append(os.path.join(planes[0],'F_mlp.npy'))
        self.probability_files.append(os.path.join(planes[0],'iscell.npy'))
        self.stat_files.append(os.path.join(planes[0],'stat.npy'))

        assert (len(self.recording_files)==1 and len(self.probability_files)==1) #Make sure there is only one file.

        self.recording_file=self.recording_files[0]
        self.probability_file=self.probability_files[0]
        self.neuron_prob=np.load(self.probability_file)
        self.neuron_prob=self.neuron_prob[:,1]
        self.traces=np.load(self.recording_file)

    def threshold_neurons(self):
        self.traces=self.traces[np.where(self.neuron_prob>self.cellthreshold),:] #Need to add threshold as attirbute
        self.traces=self.traces.squeeze()
        self.stat=np.load(self.stat_files[0],allow_pickle=True)
        self.stat=self.stat[np.where(self.neuron_prob>self.cellthreshold)] #Need to add threshold as attirbute
        return

    def __call__(self):
        super().__call__()
        self.get_s2p_outputs()
        self.threshold_neurons()
        search_path = os.path.join(self.datapath,'*.tif*')
        self.images = glob.glob(search_path)
    
    def scale_image(self,image,scalar):
        image=np.copy(image)
        image_new=image*scalar
        if image_new.max()>255:
            scalar=scalar-1
            image_new, scalar = self.scale_image(image,scalar)
        return image_new,scalar
    
    def gen_masked_image(self,image,mask_colors=[],alpha=0.2,scalar=1):
        """
        Inputs 
        mask_colors (numpy array or list) -- numbers [0,1,2...9] that is the same length of number of cells. 
        Each number will be associated with a color mask.

        Outputs
        masked_image 
        """
        # Grab coordinates for all ROIs
        self.coordinates=[]
        for i in range(len(self.stat)):
            cellx,celly=self.stat[i]['xpix'],self.stat[i]['ypix']
            self.coordinates.append([cellx,celly])

        if len(mask_colors)==0:
            # Run if we do not want any masks to be over data
            img = Image.open(image)
            img = np.asarray(img)
            img, scalar = self.scale_image(img,scalar)
            img = np.float32(img) #Convert again
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) #Convert gray scale image to color
            img = (img-img.min())/(img.max()-img.min())*255
            img = img.astype(np.uint8)
            mergedimg = img #Convert gray scale image to color

        else:
            # Assign color to cells
            mask_colors = np.asarray(mask_colors)
            if len(np.unique(mask_colors))>=9:
                raise Exception('There can only be 10 total groups for masked images. If you need greater you must edit gen_masked_image method')
            
            # Generate list of colors for each cell based on group
            colors =[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,0,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128)]
            colorlist=[]
            for group in mask_colors:
                for color in range(len(np.unique(mask_colors))):
                    if group==color:
                        colorlist.append(colors[color])

            # Open the image
            img = Image.open(image)
            img = np.asarray(img)
            img, scalar = self.scale_image(img,scalar)
            img = np.float32(img) #Convert again
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) #Convert gray scale image to color
            shapeor = img.shape

            # Add mask using colors from colorlist to image
            blank = np.zeros(shape=shapeor).astype(np.float32)
            for (cellx,celly),coloroh in zip(self.coordinates,colorlist):
                blank[celly,cellx,:]=coloroh # Set the cell's coordinates to the corresponding color

            blank= blank*alpha
            img+=np.round(blank)
            img = (img-img.min())/(img.max()-img.min())*255
            img = img.astype(np.uint8)
            mergedimg=img
            #mergedimg = cv2.addWeighted(img, alpha , blank, 1-alpha, 0) # Overlay blank image with an alpha of 0.4
        return mergedimg 

    def get_image_zoomed(self,image,neuron_id,pixel_pad=10):
        for i,(cellx,celly) in enumerate(self.coordinates):
            if i==neuron_id:
                ourx,oury=cellx,celly
        
        zoomed_image = image[(oury.min()-pixel_pad):(oury.max()+pixel_pad),(ourx.min()-pixel_pad):(ourx.max()+pixel_pad)]
        return zoomed_image    

    def create_vid_for_gui(self,neuron_id,image_number,intensity=10):
        # Apply read and optionally apply mask to current image
        if self.include_mask:
            masking = np.zeros(len(self.traces))
            #masking[neuron_id]=1 #Set current cell to different color
            masking[neuron_id]=(self.true_classification[:,0].max()+1) #Set current cell to different color
            if len(masking)<len(self.true_classification):
                masking+=1
            elif len(masking)==len(self.true_classification):
                masking+=self.true_classification[:,0] # Include previously accepted neurons
            else:
                raise Exception("shape of masking or true classification attributes are wrong")
        else:
            masking=[]
        image=self.gen_masked_image(self.corrected_images[image_number],mask_colors=masking,scalar=intensity)
        imshape=image.shape # Get height and width of image

        # Zoom in on image
        cut_image = self.get_image_zoomed(image,neuron_id=neuron_id)
        cut_image=cv2.resize(cut_image,(imshape[0],imshape[1])) # Rezie the image to make bigger

        # Get trace and morphology data for current trace on hand
        dataoh = np.copy(self.traces)
        population_activity = dataoh.mean(axis=0)
        trace_oh = dataoh[neuron_id]
        norm_trace_oh = (trace_oh-trace_oh.min())/(trace_oh.max()-trace_oh.min())*100 # Force the trace to fall between 0 and 100

        #Set up shape of window
        shape=(imshape[0]*2,imshape[1]*2,3) # Get Shape of new image
        blankimg = np.zeros(shape, np.float32) #Create background for new image
        blankimg[:-(imshape[0]),imshape[1]:imshape[1]*2]=image #Put our current masked image in
        blankimg[:-(imshape[0]),:imshape[1]]=cut_image #Put the zoomed cell image in
        #blankimg,scalar=self.scale_image(blankimg,scalar) # Change the image intensity
        
        # Zoom in on trace data currently displayed
        ys=norm_trace_oh[(image_number-49):(image_number+1)]+(imshape[0]-300)
        if ys.size==0:
            ys=norm_trace_oh[:(image_number+1)]+(imshape[0]-300)
            xs=range(0,image_number*5,5)
        else:
            xs=range(0,(image_number+1)*5,5)

        if len(xs)>1:
            xs = [(image_number*5 + 350) for image_number in range(len(ys))]
            self.b2start,self.b2stop=min(xs),max(xs)
            # Set up shape for box for later
            if len(xs)==50:
                self.bstartreal+=(1/self.skip_factor)
                self.bstopreal+=(1/self.skip_factor)
                self.bstart=round(self.bstartreal)
                self.bstop=round(self.bstopreal)
            else:
                self.bstart=0
                self.bstopreal+=(1/self.skip_factor)
                self.bstop=round(self.bstopreal)

            ys = blankimg.shape[0]-ys-50
            draw_points = (np.asarray([xs, ys]).T).astype(np.int32)
            blankimg = cv2.polylines(blankimg, [draw_points], False, (255,255,255),2)
        else:
            self.b2start,self.b2stop=350,355
            self.bstartreal=0
            self.bstart=0
            self.bstop=1
            self.bstopreal=(1/5)

        # Plot the entire trace
        draw_x,draw_y=[],[]
        self.skip_factor = round(len(norm_trace_oh)/blankimg.shape[1])
        down_sampel_trace=norm_trace_oh[::self.skip_factor]
        for xs,ys in zip(range(len(down_sampel_trace)),down_sampel_trace):
            ys=blankimg.shape[0]-ys-75
            draw_x.append(xs)
            draw_y.append(ys)
            
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        blankimg = cv2.polylines(blankimg, [draw_points], False, (255,255,255),2)
        colorimg = np.float32(blankimg)
        
        # Plot Population Acitivty
        draw_x,draw_y=[],[]
        self.skip_factor = round(len(population_activity)/blankimg.shape[1])
        population_activity=population_activity[::self.skip_factor]
        for xs,ys in zip(range(len(population_activity)),population_activity):
            ys=blankimg.shape[0]-ys+50
            draw_x.append(xs)
            draw_y.append(ys)
            
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        colorimg = cv2.polylines(colorimg, [draw_points], False, (0,255,0),2)
        colorimg=cv2.putText(colorimg, 'Normalized ROI Activity', (10,imshape[0]+50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=1,color=(255,255,255))
        colorimg=cv2.putText(colorimg, 'Population Activity', (10,imshape[0]+100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=1,color=(0,255,0))   
        colorimg=cv2.putText(colorimg, f'Neuron {neuron_id} out of {len(self.traces)}', (10,imshape[0]+150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=1,color=(0,0,255))   

        # Plot a box around current data
        boxbottom=colorimg.shape[0]-75
        boxtop=colorimg.shape[0]-180
        colorimg = cv2.rectangle(colorimg, (self.bstart, boxtop), (self.bstop, boxbottom), color=(255,0,0), thickness=2)

        # Plot a box around zoom data
        box2bottom=colorimg.shape[0]-260
        box2top=colorimg.shape[0]-366
        colorimg = cv2.rectangle(colorimg, (self.b2start, box2top), (self.b2stop, box2bottom), color=(255,0,0), thickness=2)

        #Draw lines
        colorimg = cv2.line(colorimg, (self.bstart, boxtop), (self.b2start, box2bottom), color=(255,0,0), thickness=2)
        colorimg = cv2.line(colorimg, (self.bstop, boxtop), (self.b2stop, box2bottom), color=(255,0,0), thickness=2)

        colorimg=cv2.resize(colorimg,(1000,1000))
        return colorimg

if __name__=='__main__':
    ev_obj=manual_classification(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058')
    ev_obj()