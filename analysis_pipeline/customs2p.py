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
# import PySimpleGUI as psg
sns.set_style('whitegrid')

class get_s2p():
    """ get suite 2P:
    This class is meant to run suite2P without the gui. 
    """
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1):
        #Set input and output directories
        self.datapath=datapath
        self.resultpath=os.path.join(self.datapath,'figures/')
        self.resultpath_so=os.path.join(self.datapath,'figures/serialoutput/')
        self.resultpath_neur=os.path.join(self.datapath,'figures/neuronal/')

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
        self.ops['input_format']="bruker"
        self.ops['blocksize']=blocksize
        self.ops['reg_tif']=reg_tif
        self.ops['reg_tif_chan2']=reg_tif_chan2
        self.ops['denoise']=denoise

        #Set up datapath
        self.db = {'data_path': [self.datapath],}
    
    def __call__(self):
        searchstring=os.path.join(self.datapath,'**/F.npy')
        res = glob.glob(searchstring,recursive=True)
        if not res:
            self.auto_run()
            self.get_reference_image()

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

class manual_classification(get_s2p):
    def get_s2p_outputs(self):
        #Find planes and get recording/probability files
        search_path = os.path.join(self.datapath,'suite2p/plane*/')
        self.recording_files=[]
        self.probability_files=[]
        self.stat_files=[]
        planes = [result for result in glob.glob(search_path)]
        self.recording_files.append(os.path.join(planes[0],'F.npy'))
        self.probability_files.append(os.path.join(planes[0],'iscell.npy'))
        self.stat_files.append(os.path.join(planes[0],'stat.npy'))

        assert (len(self.recording_files)==1 and len(self.probability_files)==1) #Make sure there is only one file.

        self.recording_file=self.recording_files[0]
        self.probability_file=self.probability_files[0]
        self.neuron_prob=np.load(self.probability_file)
        self.neuron_prob=self.neuron_prob[:,1]
        self.traces=np.load(self.recording_file)

    def threshold_neurons(self):
        self.traces=self.traces[np.where(self.neuron_prob>0.9),:] #Need to add threshold as attirbute
        self.traces=self.traces.squeeze()
        return

    def __call__(self):
        self.get_s2p_outputs()
        self.threshold_neurons()
        self.stat=np.load(self.stat_files[0],allow_pickle=True)
        search_path = os.path.join(self.datapath,'*.tif*')
        self.images = glob.glob(search_path)
    
    def scale_image(self,image,scalar):
        image=np.copy(image)
        image_new=image*scalar
        if image_new.max()>255:
            scalar=scalar-1
            image_new = self.scale_image(image,scalar)
        return image_new,scalar

    def create_vids(self,neuron_id,image_number):
        # Get min and max values of all ROIs
        for i in range(len(self.stat)):
            cellx,celly=self.stat[i]['xpix'],self.stat[i]['ypix']
            if i==0:
                frx0,frx1=cellx.min()-10,cellx.max()+10
                fry0,fry1=celly.min()-10,celly.max()+10
            else: 
                if (cellx.min())<frx0:
                    frx0=cellx.min()
                if (cellx.max())>frx1:
                    frx1=cellx.min()
                if (celly.min())<fry0:
                    fry0=celly.min()
                if (celly.max())>fry1:
                    fry1=celly.min()
        fullcrop=[frx0,frx1,fry0,fry1]

        # Get trace and morphology data for current trace on hand
        dataoh = np.copy(self.traces)
        trace_oh = dataoh[neuron_id]
        cellx,celly=self.stat[neuron_id]['xpix'],self.stat[neuron_id]['ypix']
        frx0,frx1=cellx.min()-10,cellx.max()+10
        fry0,fry1=celly.min()-10,celly.max()+10
        norm_trace_oh = (trace_oh-trace_oh.min())/(trace_oh.max()-trace_oh.min())*100

        scalar=40 #similar to setting intensity
        image=self.images[image_number] 
        img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        inith,initw=img.shape[0],img.shape[1]
        img_crop = img

        #Cut out the specific cell
        cut_image=img[frx0:frx1,fry0:fry1]
        cut_image=cv2.resize(cut_image,(inith,initw))


        #Set up shape of window
        shape=(inith*2,initw*2)
        blankimg = np.zeros(shape, np.float64)
        blankimg[:-(inith),initw:initw*2]=img_crop 
        blankimg[:-(inith),:initw]=cut_image
        blankimg=blankimg/255
        blankimg,scalar=self.scale_image(blankimg,scalar)
        
        # Zoom in on trace data currently displayed
        ys=norm_trace_oh[(image_number-49):(image_number+1)]+(inith-300)
        if ys.size==0:
            ys=norm_trace_oh[:(image_number+1)]+(inith-300)
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

        blankimg = np.float32(blankimg)
        colorimg = cv2.cvtColor(blankimg,cv2.COLOR_GRAY2RGB)
        
        # #Add in mask data
        for xc,yc in zip(cellx,celly):
            xc+=initw
            b,g,r=colorimg[xc,yc,:]
            colorimg[yc,xc,:]=[b,g,r+10]
        

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
    ev_obj=manual_classification(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620083_M3_R1-052')
    ev_obj()