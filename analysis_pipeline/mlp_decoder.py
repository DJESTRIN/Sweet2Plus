from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
import ipdb
from keras.regularizers import l2
import optuna
import optkeras
optkeras.optkeras.get_trial_default = lambda: optuna.trial.FrozenTrial(None, None, None, None, None, None, None, None, None, None, None)
from optkeras.optkeras import OptKeras


ok = OptKeras(study_name='neural_encoder')

def objective(trial):
    #Load data
    X=np.load('X.npy')
    y=np.load('y.npy')

    #Convert to one hot vector
    y_onehot=np.zeros((y.shape[0],4))
    for i,j in enumerate(y):
        y_onehot[i,j]=1

    # Get stats on data
    input_dimoh=X.shape[1]
    output_dim=4
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2,shuffle=True,stratify=y_onehot)

    # Clear backend
    keras.backend.clear_session()   

    # Create model
    model_mlp = Sequential()
    model_mlp.add(keras.Input(shape=(1214,)))
    model_mlp.add(Dense(50 , kernel_initializer="uniform", activation = 'relu', kernel_regularizer=l2(trial.suggest_float('l2', low=0, high=0.3))))
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(30, kernel_initializer="uniform", activation = 'relu', kernel_regularizer=l2(0.01)))
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(output_dim, activation = 'softmax'))
    model_mlp.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model_mlp.fit(X_train, y_train,validation_data = (X_test, y_test), batch_size = trial.suggest_int('batch_size', low=4, high=100), 
                            epochs = 100, shuffle=True,callbacks = ok.callbacks(trial), verbose = ok.keras_verbose)

    return ok.trial_best_value

ok.optimize(objective, timeout = 600) # 1 minute for demo
