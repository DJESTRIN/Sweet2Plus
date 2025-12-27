import torch
import numpy as np
import glob,os
import ipdb
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import matplotlib.cm as cm
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
        self.outputs = []
        self.probabilities = []

        for trace in tqdm.tqdm(self.normtraces):
            input_tensor = torch.tensor(trace).double().to(device=device)
            output = net(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=0)
            pred_class = torch.argmax(probs, dim=0)
            self.outputs.append(pred_class.cpu().detach().numpy())
            self.probabilities.append(probs.cpu().detach().numpy())

        self.outputs = np.asarray(self.outputs)
        self.probabilities = np.asarray(self.probabilities)
    
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
        # Randomly pull 16 neurons from each
        pulled_real = np.random.randint(0, self.real_traces.shape[0], 16)
        pulled_noise = np.random.randint(0, self.noise.shape[0], 16)

        # Stack the sampled traces to compute combined min and max
        sampled_real = self.real_traces[pulled_real, :]
        sampled_noise = self.noise[pulled_noise, :]
        all_sampled = np.vstack([sampled_real, sampled_noise])

        # Compute y-limits based only on the sampled neurons
        ymin = all_sampled.min()
        ymax = all_sampled.max()

        # --- Plot real traces ---
        fig = plt.figure(figsize=(10, 10))
        for i, k in enumerate(pulled_real):
            plt.subplot(4, 4, i + 1)
            plt.plot(self.real_traces[k, :], color='seagreen')
            plt.ylim([ymin, ymax])
        fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
        fig.text(0.06, 0.5, 'Suite2P Extracted dF', ha='center', va='center', rotation='vertical')
        plt.savefig('RealSignal.jpg')
        plt.close(fig)

        # --- Plot noise traces ---
        fig = plt.figure(figsize=(10, 10))
        for i, k in enumerate(pulled_noise):
            plt.subplot(4, 4, i + 1)
            plt.plot(self.noise[k, :], color='firebrick')
            plt.ylim([ymin, ymax])
        fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
        fig.text(0.06, 0.5, 'Suite2P Extracted dF', ha='center', va='center', rotation='vertical')
        plt.savefig('Noise.jpg')
        plt.close(fig)

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
        all_traces = []
        all_probs = []
        for file in self.Ffiles:
            mlpoh=MLPapply(model_path=self.model_path,data_path=file,plot_examples=True)
            real_traces,noise=mlpoh()
            all_traces.append(mlpoh.traces)
            all_probs.append(mlpoh.probabilities)
            self.all_real.append(real_traces)
            self.all_noise.append(noise)
        
        all_traces_np = np.vstack(all_traces)
        all_probs_np = np.vstack(all_probs)
        sort_idx = np.argsort(all_probs_np[:, 1])[::-1]

        sorted_traces = all_traces_np[sort_idx]
        sorted_probs = all_probs_np[sort_idx]

        #############
        
        def calculate_bandwidth(traces):
            return traces.max(axis=1) - traces.min(axis=1)

        # Masks for high and low probability neurons
        high_mask = sorted_probs[:, 1] > 0.9
        low_mask = sorted_probs[:, 1] < 0.1

        # Extract traces for each group
        high_traces = sorted_traces[high_mask]
        low_traces = sorted_traces[low_mask]

        # Calculate bandwidth for each group
        high_bandwidths = calculate_bandwidth(high_traces)
        low_bandwidths = calculate_bandwidth(low_traces)

        # Compute median bandwidth for each group
        high_median_bw = np.median(high_bandwidths)
        low_median_bw = np.median(low_bandwidths)

        # Filter traces based on median bandwidth
        high_keep_mask = high_bandwidths >= high_median_bw
        low_keep_mask = low_bandwidths >= low_median_bw

        high_traces_filtered = high_traces[high_keep_mask]
        low_traces_filtered = low_traces[low_keep_mask]

        # Check if enough traces remain
        n_sample = 25
        if high_traces_filtered.shape[0] < n_sample or low_traces_filtered.shape[0] < n_sample:
            raise ValueError(f"Not enough traces after filtering: "
                            f"{high_traces_filtered.shape[0]} high prob, "
                            f"{low_traces_filtered.shape[0]} low prob. "
                            f"Try lowering probability thresholds or adjust sampling.")

        # Randomly sample from filtered traces
        high_sample_idx = np.random.choice(high_traces_filtered.shape[0], size=n_sample, replace=False)
        low_sample_idx = np.random.choice(low_traces_filtered.shape[0], size=n_sample, replace=False)

        final_high_traces = high_traces_filtered[high_sample_idx]
        final_low_traces = low_traces_filtered[low_sample_idx]

        # Corresponding probabilities for titles
        final_high_probs = sorted_probs[high_mask][high_keep_mask][high_sample_idx, 1]
        final_low_probs = sorted_probs[low_mask][low_keep_mask][low_sample_idx, 1]

        # Combine traces and probs
        all_traces = np.vstack((final_high_traces, final_low_traces))
        all_probs = np.concatenate((final_high_probs, final_low_probs))

        # Compute y limits robustly (1st and 99th percentile)
        ymin, ymax = np.min(all_traces), np.max(all_traces)
        # Plot in 5x10 grid
        fig, axes = plt.subplots(5, 10, figsize=(20, 10), sharey=True, sharex=True)

        for i, ax in enumerate(axes.flatten()):
            trace = all_traces[i]
            prob = all_probs[i]
            color = 'seagreen' if i < n_sample else 'firebrick'
            ax.plot(trace, color=color)
            ax.set_title(f"P(real)={prob:.2f}", fontsize=8)
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([])
            ax.set_yticks([])

        fig.text(0.5, 0.04, 'Frames', ha='center')
        fig.text(0.06, 0.5, 'Normalized dF', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.07, 0.07, 1, 0.95])
        plt.savefig("TraceGrid_HighLow_MedianBandwidthFiltered.jpg")
        plt.close(fig)

        # Initialize list to hold all rows
        rows = []

        # Assign neuron IDs starting from 0
        neuron_id = 0

        # Loop over high-probability traces
        for trace, prob in zip(final_high_traces, final_high_probs):
            for t, val in enumerate(trace):
                rows.append({
                    'neuron': neuron_id,
                    'time': t,
                    'activity': val,
                    'classification': 1,
                    'probability': prob
                })
            neuron_id += 1

        # Loop over low-probability traces
        for trace, prob in zip(final_low_traces, final_low_probs):
            for t, val in enumerate(trace):
                rows.append({
                    'neuron': neuron_id,
                    'time': t,
                    'activity': val,
                    'classification': 0,
                    'probability': prob
                })
            neuron_id += 1

        # Create DataFrame
        trace_df = pd.DataFrame(rows)

        # Save to CSV
        trace_df.to_csv("trace_data_long.csv", index=False)

        # Example input: sorted_traces = np.array(...) with shape (n_cells, n_timepoints)
        n = sorted_traces.shape[0]
        group_size = n // 10  # 10 percentile bins

        # Storage for export
        rows = []

        # Colormap for plotting (optional)
        colors = cm.viridis(np.linspace(0, 1, 10))

        for i in range(10):
            start = i * group_size
            end = (i + 1) * group_size if i < 9 else n  # Last bin takes the rest
            group_traces = sorted_traces[start:end]
            group_mean = group_traces.mean(axis=0)
            group_sem = sem(group_traces, axis=0)

            percentile_label = f"{(100 - (i + 1)*10)}-{100 - i*10}"

            for frame_idx, (mean_val, sem_val) in enumerate(zip(group_mean, group_sem)):
                rows.append({
                    "percentile_bin": percentile_label,
                    "frame": frame_idx,
                    "mean": mean_val,
                    "sem": sem_val
                })

        # Create DataFrame and save
        df_export = pd.DataFrame(rows)
        df_export.to_csv("trace_percentile_summary.csv", index=False)

        ipdb.set_trace()

    
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

        

     