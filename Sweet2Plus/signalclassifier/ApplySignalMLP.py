import torch
import numpy as np
import glob,os
import ipdb
import tqdm
import matplotlib.pyplot as plt
from Sweet2Plus.signalclassifier.SignalMLP import downsample_array, normalize_trace, NeuralNetwork

class MLPapply():
    def __init__(self,model_path,data_path,plot_examples=False):
        self.model=torch.load(model_path,map_location=torch.device('cpu')) #Load model as attribute
        self.open_data(data_path=data_path)
        self.plot_examples=plot_examples

    def downsample_array(self,X):
        return downsample_array(X)

    def open_data(self,data_path):
        """ inputs: data_path -- full path to F.npy file produced by suite2p """
        self.traces=np.load(data_path) #orginal traces
        self.dstraces=self.downsample_array(self.traces) # downsampled traces
        pseudodata=np.ones(shape=(self.dstraces.shape[0],)) #This can be ignored, acts as a place holder
        self.normtraces,_=normalize_trace(traces=self.dstraces,labels=pseudodata) #normalized data
    
        #set up output path
        output_path=os.path.dirname(data_path)
        self.output_file=os.path.join(output_path,'F_mlp.npy')
    
    def __call__(self):
        self.run_model()
        self.sep_traces()
        self.post_process()
        if self.plot_examples:
            self.plot_outputs()
        self.save_real()
        return self.real_traces, self.noise

    def run_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        net = NeuralNetwork()
        net.load_state_dict(self.model)
        net=net.to(device=device).double()
        self.outputs=[]
        for trace in tqdm.tqdm(self.normtraces):
            output = net(torch.tensor(trace).double().to(device=device))
            output=torch.argmax(output, dim = 0)
            self.outputs.append(output.cpu().detach().numpy())
        self.outputs=np.asarray(self.outputs)
    
    def post_process(self):
        """ Manually clean up =>
        remove any accidental traces that are a flat line.  
        """
        self.corrected_traces=[]
        for trace in self.real_traces:
            if np.mean(trace)==0:
                continue
            else:
                self.corrected_traces.append(trace)
        
        self.real_traces=np.asarray(self.corrected_traces)

    def sep_traces(self):
        self.real_traces=self.traces[self.outputs==1,:]
        self.noise=self.traces[self.outputs==0,:]

    def plot_outputs(self):
        # Generates Figure of Traces classified as real neurons
        pulled = np.random.randint(0,self.real_traces.shape[0],16)
        fig=plt.figure(figsize=(10,10))
        for i,k in enumerate(pulled):
            plt.subplot(4,4,i+1)
            plt.plot(self.real_traces[k,:],color='seagreen')
        fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
        fig.text(0.06, 0.5, 'Suite2P Extracted dF', ha='center', va='center', rotation='vertical')
        plt.savefig('RealSignal.jpg')

        # Generate Figure of Traces classified as noise
        pulled = np.random.randint(0,self.noise.shape[0],16)
        plt.figure(figsize=(10,10))
        for i,k in enumerate(pulled):
            plt.subplot(4,4,i+1)
            plt.plot(self.noise[k,:],color='firebrick')
        fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
        fig.text(0.06, 0.5, 'Suite2P Extracted dF', ha='center', va='center', rotation='vertical')
        plt.savefig('Noise.jpg')

    def save_real(self):
        np.save(self.output_file,self.real_traces)

class RunMLPFull():
    def __init__(self,model_path,search_string):
        self.model_path=model_path
        self.search_string=search_string
        self.all_real=[]
        self.all_noise=[]

    def find_files(self):
        #get all the F.npy files in folder of interest
        self.Ffiles=glob.glob(self.search_string)

    def run_classification(self):
        #run the MLP on each found file
        for file in self.Ffiles:
            mlpoh=MLPapply(model_path=self.model_path,data_path=file,plot_examples=False)
            real_traces,noise=mlpoh()
            self.all_real.append(real_traces)
            self.all_noise.append(noise)
    
    def quickplot_all(self):
        #plot all real traces and fake
        plt.figure(figsize=(5,40),dpi=300)
        spacer=0
        counter=0
        for subject in self.all_real:
            subject=np.asarray(subject)
            for trace in subject:
                if counter%100==0:
                    traceoh=((trace-np.min(trace))/(np.max(trace)-np.min(trace)))+spacer
                    plt.plot(traceoh,color='black')
                    spacer+=1
                counter+=1
        
        plt.savefig('all_real_traces.pdf')
        plt.close()

        plt.figure(figsize=(5,40),dpi=300)
        spacer=0
        counter=0
        for subject in self.all_noise:
            subject=np.asarray(subject)
            for trace in subject:
                if counter%100==0:
                    try:
                        traceoh=((trace-np.min(trace))/(np.max(trace)-np.min(trace)))+spacer
                    except:
                        traceoh=trace+spacer
                    plt.plot(traceoh,color='black')
                    spacer+=1
                counter+=1
        
        plt.savefig('all_noise_traces.pdf')
        plt.close()

if __name__=='__main__':
    data=r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\**\*24*\*24*\suite2p\*plane*\F.npy'
    model=r'C:\Users\listo\twophoton\analysis_pipeline\best_model_weights.pth'
    MLPs=RunMLPFull(model_path=model,search_string=data)
    MLPs.find_files()
    MLPs.run_classification()
    MLPs.quickplot_all()

        

     