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
from torch import nn
from torch.utils.data import DataLoader
import optuna
import pandas as pd
from datetime import datetime
import re

def downsample_array(array,new_length=1000):
    array_new=np.zeros((array.shape[0],new_length))
    length=array.shape[1]
    skip=int(np.ceil(length/new_length))
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

def Average(lst):
    return sum(lst) / len(lst)

def TrainTestNetwork(Network, train_loader, test_loader, learningrate, batchsize, epochs, 
                     xls_drop_directory=r'C:\tmt_assay\tmt_experiment_2024_clean\mlp_roi_classifier_results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    
    best_f1 = 0
    training_results = []
    testing_results = []
    all_epoch_metrics = []  # store metrics per epoch for Excel

    for epoch in range(epochs):
        # Reset optimizer per epoch (like original code)
        optimizer = torch.optim.SGD(Network.parameters(), lr=learningrate)
        
        # --- Training ---
        Network.train()
        f1_train_av = []
        ac_train_av = []
        losses = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = Network(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            losses.append(train_loss)
            outputs_argmax = torch.argmax(outputs, dim=1)
            f1_train_av.append(f1_score(labels.cpu().detach().numpy(), outputs_argmax.cpu().detach().numpy()))
            ac_train_av.append(accuracy_score(labels.cpu().detach().numpy(), outputs_argmax.cpu().detach().numpy()))
        
        # --- Testing ---
        Network.eval()
        f1_test_av = []
        ac_test_av = []
        test_cm_list = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Network(inputs.float())
            outputs_argmax = torch.argmax(outputs, dim=1)
            f1_test_av.append(f1_score(labels.cpu().detach().numpy(), outputs_argmax.cpu().detach().numpy()))
            ac_test_av.append(accuracy_score(labels.cpu().detach().numpy(), outputs_argmax.cpu().detach().numpy()))
            cm = confusion_matrix(labels.cpu().detach().numpy(), outputs_argmax.cpu().detach().numpy())
            test_cm_list.append(cm.flatten())
        
        # Average metrics per epoch
        train_f1_avg = np.mean(f1_train_av)
        train_acc_avg = np.mean(ac_train_av)
        train_loss_avg = np.mean(losses)
        test_f1_avg = np.mean(f1_test_av)
        test_acc_avg = np.mean(ac_test_av)
        test_cm_avg = np.mean(test_cm_list, axis=0) if len(test_cm_list) > 0 else np.array([np.nan]*4)
        
        # Save best model
        if test_f1_avg > best_f1:
            best_f1_on_disk = -1.0
            if os.path.exists(xls_drop_directory):
                for fname in os.listdir(xls_drop_directory):
                    match = re.match(r"best_model_f1_([0-9.]+)\.pth", fname)
                    if match:
                        best_f1_on_disk = float(match.group(1))
                        break

            if test_f1_avg > best_f1_on_disk:
                best_f1 = test_f1_avg
                best_model_weights = os.path.join(xls_drop_directory,f"best_model_f1_{best_f1:.4f}.pth")
                torch.save(Network.state_dict(), best_model_weights)
                print(f"Epoch {epoch}: Saved best model "
                      f"(Test F1 = {best_f1:.4f}, Test Accuracy = {test_acc_avg:.4f})")
        
        # Decay learning rate every 100 epochs (like original)
        if epoch % 100 == 0 and epoch != 0:
            learningrate *= 0.1

        # Store results for plotting
        training_results.append(train_f1_avg)
        testing_results.append(test_f1_avg)
        
        # Save metrics for Excel
        all_epoch_metrics.append({
            'epoch': epoch,
            'train_f1': train_f1_avg,
            'train_accuracy': train_acc_avg,
            'train_loss': train_loss_avg,
            'test_f1': test_f1_avg,
            'test_accuracy': test_acc_avg,
            'test_cm_00': test_cm_avg[0],
            'test_cm_01': test_cm_avg[1],
            'test_cm_10': test_cm_avg[2],
            'test_cm_11': test_cm_avg[3],
            'learning_rate': learningrate,
            'batch_size': batchsize
        })
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Train F1={train_f1_avg:.4f}, Test F1={test_f1_avg:.4f}, Train Loss={train_loss_avg:.6f}, LR={learningrate:.6f}')

    # Save Data to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = os.path.join(xls_drop_directory, f"training_metrics_{timestamp}.xlsx")
    df_results = pd.DataFrame(all_epoch_metrics)
    df_results.to_excel(output_excel, index=False)
    
    # Return results like original function
    testing_results_np = np.asarray(testing_results)
    f1_test_average = testing_results_np.max()
    print(f'Max Testing F1: {f1_test_average}')

    return df_results


def main(search_string,lr,bs):
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
    results_oh = TrainTestNetwork(net, train_dataloader, test_dataloader, lr, bs, 1000)
    return results_oh, net

def objective(trial):
    """ Generate Hyperparmeters """
    #Optuna Derived Hyperparameters
    Learing_Rate = trial.suggest_float('Learing_Rate', 0.45, 0.47, log=True) #Initial learning rate
    Batch_Size = trial.suggest_int('Batch_Size', 160, 170) #Batch Size

    results_oh,model=main(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day1\*24*\*24*\suite2p\*lane*',Learing_Rate,Batch_Size)
    f1_test_average = results_oh['test_f1'].mean()
    return f1_test_average

if __name__=='__main__':
    choice = input("Do you want to run an Optuna study or use manual hyperparameters? (optuna/manual): ").strip().lower()
    if choice == "optuna":
        study = optuna.create_study(study_name='SignalMLP_e1', direction='maximize')
        study.optimize(objective, n_trials=100)
        optuna.visualization.plot_optimization_history(study)
        
    elif choice == "manual":
        results_oh, model=main(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day1\*24*\*24*\suite2p\*lane*', 0.46859501736598963,170)
    else:
        print("Invalid choice. Please enter 'optuna' or 'manual'.")
