import torch
import numpy as np
import glob,os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import ipdb
import optuna
import tqdm
import matplotlib.pyplot as plt

def downsample_array(array,new_length=1000):
    array_new=np.zeros((array.shape[0],new_length))
    length=array.shape[1]
    skip=round(length/new_length)
    array_oh=array[:,::skip]
    array_new[:array_oh.shape[0],:array_oh.shape[1]]=array_oh
    return array_new

def generate_data(search_sting):
    subjects=glob.glob(search_sting)
    for i,subject in enumerate(subjects):
        cellfile=os.path.join(subject,'iscell.npy')
        tracefile=os.path.join(subject,'F.npy')
        booleans=np.load(cellfile)
        traces=np.load(tracefile)
        traces=downsample_array(traces)
        if i==0:
            x=traces
            y=booleans[:,0]
        else:
            x=np.concatenate((x,traces), axis=0)
            y=np.concatenate((y,booleans[:,0]), axis=0)
    return x,y

def normalize_trace(traces,labels):
    normalize_traces=[]
    norm_lables=[]
    for index,(trace,label) in enumerate(zip(traces,labels)):
        ntrace=(trace-trace.min())/(trace.max()-trace.min()) #force traces to fall between 0 and 1. 
        if np.isnan(ntrace).any():
            if label==1:
                label=0
            ntrace=np.zeros(ntrace.shape)
        normalize_traces.append(ntrace)
        norm_lables.append(label)
    normalize_traces=np.asarray(normalize_traces)
    return normalize_traces,norm_lables

def expand_training_dataset(X,Y):
    expanded_trainingdata,expanded_labels=[],[]
    random_trace_for_plot=np.random.randint(low=0,high=X.shape[0],size=10)
    for toh,(trace,label) in enumerate(zip(X,Y)):
        plot_data=[]
        for k in range(3): # Add 0 to 10 percent noise to trace 
            scalar=(k/100)*7
            noise_trace = (np.random.uniform(0,1,1000))*scalar
            new_trace=trace+noise_trace
            expanded_trainingdata.append(new_trace)
            expanded_labels.append(label)
            if toh in random_trace_for_plot:
                plot_data.append(new_trace)

        if toh in random_trace_for_plot:
           plt.figure()
           for r,plottrace in enumerate(plot_data):
               plt.subplot(3, 1, r+1)
               plt.plot(plottrace)
           plt.savefig(f'Noisytraces{toh}.jpg')
           plt.close()
    expanded_trainingdata=np.asarray(expanded_trainingdata)
    expanded_labels=np.asarray(expanded_labels)
    return expanded_trainingdata,expanded_labels

class Load_Data(Dataset):
    def __init__(self, traces, classifications, device):
        #Load in numpy data (data_set, responses) into pytorch dataclass
        self.traces=torch.tensor(traces)
        classifications=classifications.astype(np.uint8)
        self.classifications=torch.tensor(classifications)

        #Put it on device
        self.traces = self.traces.to(device)
        self.classifications = self.classifications.to(device)

    def __len__(self):
        return len(self.classifications)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        output = self.classifications[idx]
        return trace, output

#Standard MLP
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        res = self.linear_relu_stack(x)
        return res

def Diagnostics(epoch, loss, outputs, labels, learningrate, batch_size, data_set_name):
    predictions = outputs
    accuracy = accuracy_score(predictions, labels)
    f1 = f1_score(predictions,labels)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    Diagnostics_list = [epoch, data_set_name, loss, learningrate, batch_size, accuracy, f1, tn, fp, fn, tp]
    return Diagnostics_list, f1

def Average(lst):
    return sum(lst) / len(lst)

def TrainTestNetwork(Network, train_loader, test_loader, learningrate, batch_size, epochs):
    # Set up optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Network.parameters(), lr = learningrate)

    # Run training and testing
    training_results,testing_results=[],[]
    for epoch in range(epochs):
        optimizer = torch.optim.SGD(Network.parameters(), lr = learningrate)
        # Train Neural Network 
        f1_train_av = []
        ac_train_av =[]
        losses=[]
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = Network(inputs.float())
            loss = criterion(outputs, labels) #Not great but prob works
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(Network.parameters(), 5)
            optimizer.step()
            train_loss = loss.item()
            losses.append(train_loss)
            outputs = torch.argmax(outputs,dim=1)
            f1_train_av.append(f1_score(labels.cpu().detach().numpy(),outputs.cpu().detach().numpy())) 
            ac_train_av.append(accuracy_score(labels.cpu().detach().numpy(),outputs.cpu().detach().numpy())) 

        # Test Neural Network
        f1_test_av = []
        ac_test_av = []
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = Network(inputs.float())
            outputs = torch.argmax(outputs, dim = 1)
            f1_test_av.append(f1_score(labels.cpu().detach().numpy(),outputs.cpu().detach().numpy())) 
            ac_test_av.append(accuracy_score(labels.cpu().detach().numpy(),outputs.cpu().detach().numpy())) 
        
        #Save and Print Results
        if epoch%10==0:
            print('Epoch [%d] Training loss: %.8f Training F1 Score: %.8f Training Accuracy: %.8f Testing F1 Score: %.8f Testing Accuracy: %.8f Learning Rate: %.8f' % (epoch, Average(losses), Average(f1_train_av), Average(ac_train_av), Average(f1_test_av), Average(ac_test_av),learningrate))

        if epoch%100==0:
            learningrate*=0.1

        training_results.append(Average(f1_train_av))
        testing_results.append(Average(f1_train_av))
    testing_results_np=np.asarray(testing_results)
    f1_test_average=testing_results_np.max()
    print(f'Max Testing F1: {f1_test_average}')
    return training_results, testing_results,f1_test_average


def main(search_string,lr,mom,wd,bs):
    # Gather Data and Normalize it
    X,Y=generate_data(search_sting=search_string)
    X,Y=normalize_trace(X,Y)

    # Load data into a pytorch format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.15)
    X_train,y_train=expand_training_dataset(X_train,y_train)
    train_dataset = Load_Data(np.array(X_train), np.array(y_train), device)
    test_dataset = Load_Data(np.array(X_test), np.array(y_test), device)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    #Create network
    net=NeuralNetwork().to(device=device)

    # Run training regimine
    training_results,testing_results,f1_test_average = TrainTestNetwork(net, train_dataloader, test_dataloader, lr, bs,1000)
    return training_results,testing_results,f1_test_average

def objective(trial):
    """ Generate Hyperparmeters """
    #Optuna Derived Hyperparameters
    Learing_Rate = trial.suggest_float('Learing_Rate', 1e-6, 1, log=True) #Initial learning rate
    Batch_Size = trial.suggest_int('Batch_Size', 8, 264) #Batch Size

    training_results,testing_results,f1_test_average=main(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day1\*24*\*24*\suite2p\*lane*',Learing_Rate,1,1,Batch_Size)
    return f1_test_average

if __name__=='__main__':
    study = optuna.create_study(study_name='SignalMLP', direction='maximize')
    study.optimize(objective, n_trials=100)
    optuna.visualization.plot_optimization_history(study)
    #training_results,testing_results,f1_test_average=main(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day1\*24*\*24*\suite2p\*lane*',1,0.9,0.0000001,64)
    
    #training_results,testing_results=main(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day1\*24*\*24*\suite2p\*lane*')
    ipdb.set_trace()
