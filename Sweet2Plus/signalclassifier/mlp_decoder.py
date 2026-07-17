from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
from keras.regularizers import l2
import sys
import types
import optuna

# optkeras (unmaintained since 2019) relies on np.Inf, which was removed in
# NumPy 2.0 (renamed to np.inf), and on optuna.structs (see below).
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

# optkeras imports `optuna.structs`, a module that was removed in
# optuna>=2.0 (its contents were reorganized into
# optuna.trial/optuna.study/optuna.exceptions). Provide a thin compatibility
# shim so optkeras keeps working with modern optuna versions instead of
# requiring a pinned, hard-to-build ancient optuna release.
if not hasattr(optuna, 'structs'):
    _structs = types.ModuleType('optuna.structs')
    _structs.TrialState = optuna.trial.TrialState
    _structs.FrozenTrial = optuna.trial.FrozenTrial
    _structs.StudyDirection = optuna.study.StudyDirection
    _structs.StudySummary = optuna.study.StudySummary
    _structs.TrialPruned = optuna.exceptions.TrialPruned
    optuna.structs = _structs
    sys.modules['optuna.structs'] = _structs

import optkeras.optkeras
optkeras.optkeras.get_trial_default = lambda: optuna.trial.FrozenTrial(None, None, None, None, None, None, None, None, None, None, None)
from optkeras.optkeras import OptKeras


def build_objective(ok):
    """Create the Optuna objective function bound to a given OptKeras study."""
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
    return objective


def main():
    ok = OptKeras(study_name='neural_encoder')
    ok.optimize(build_objective(ok), timeout = 600) # 1 minute for demo


if __name__ == '__main__':
    main()

