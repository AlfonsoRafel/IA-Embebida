import argparse
from os.path import join
import numpy as np
from utils import WaveDataset
from utils import extract_features

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from os.path import join, isfile
from os import listdir
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib



def run(data, event_list, fs, t):
    # load model
    model_path = './models/model.pkl'
    expdir = './predict'
    model = joblib.load('model.pkl')
    # load data
    output_list = []
    for event in range(0, len(event_list)):
        start = event_list[event][0]
        end =  event_list[event][1]
        #data = data / np.iinfo(data.dtype).max
        data_raw = data[np.where((t<end) & (t>=start))] 
        
        #data = data_raw.astype(np.float32)
        X = extract_features(data_raw, fs)
        X = np.array(X)
        X = X.reshape(1, len(X))
        Y = model.predict(X)
        output_list.append(Y[0])
    return output_list

def predict(datadir):
    # load model
    model_path = './models/model.pkl'
    expdir = './predict'
    model = joblib.load('model.pkl')
    # load data
    filenames = [f for f in listdir(datadir) if (isfile(join(datadir, f)) and (f.endswith('.wav')))]
    output_list = []
    for file in filenames:
        fs, data_raw = wavfile.read(join(datadir, file), False)
        data = data_raw / np.iinfo(data_raw.dtype).max
        #data = data_raw.astype(np.float32)
        X = extract_features(data, fs)
        X = np.array(X)
        X = X.reshape(1, len(X))
        Y = model.predict(X)
        output_list.append(Y[0])

    print('Predictions made!')   
    columns = ['Filename','Class']
    output_pd = pd.DataFrame(columns=columns)
    output_pd['Filename'] = filenames
    output_pd['Class'] = output_list
    output_pd.to_csv(join(expdir, 'classification/out.csv'), index=False) 
    print('Classification File Saved')    
    
        

def train(datadir):
    expdir = './models'
    filedir = join(datadir, 'train_labels.csv')
    audiodir = join(datadir,'audios')

    train_dataset = WaveDataset(audiodir, filedir)
    print('Dataset Loaded')
    
    X_train = np.vstack([extract_features(wavdata.data, wavdata.sample_freq) for wavdata in train_dataset])
    y_train = np.array([wavdata.label for wavdata in train_dataset])    
    print('Features Extracted')

    
    X_train_opt, X_optim, y_train_opt, y_optim = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.33,
                                                              random_state=42)
    
    # Models dict
    models = {}
    sc_best = 0
    
    models['DecisionTreeClassifier'] = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
    models['MLPClassifier'] = MLPClassifier(random_state=42, max_iter=3000, early_stopping=False, n_iter_no_change=100)
    models['RandomForestClassifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for model_name, model in models.items():
        
        pipeline = imb_make_pipeline(
                              RandomUnderSampler(random_state=42),
                              StandardScaler(),
                              model,
                             )
        # pipeline = make_pipeline(
        #                   StandardScaler(),
        #                   model,
        #              )

        # Se entrena el modelo
        pipeline.fit(X_train_opt, y_train_opt)

        # Se prueba con la partición de optimización
        y_pred_optim = pipeline.predict(X_optim)

        sc = balanced_accuracy_score(y_optim, y_pred_optim)
        print(f"{sc:.3f} balanced accuracy")
        
        if sc > sc_best:
            model_best = pipeline
            print('Best Model Selected: ', model_name)
            sc_best = sc
    
        # Reporte de clasificación
        # report = classification_report(y_optim,
        #                        y_pred_optim, 
        #                        target_names=["bite", "chew", "chewbite"])

        # print(report)
        # cm = confusion_matrix(y_optim, y_pred_optim)
        # ConfusionMatrixDisplay(cm,
        #                display_labels=["bite", "chew", "chewbite"]).plot()
        # plt.grid(False)
        # plt.show()
    
    joblib.dump(model_best, 'model.pkl')


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'datadir',
        metavar='path',
        default='None',
        help='Data root to be processed')
    args = arg_parser.parse_args()
    
    predict(args.datadir)


if __name__ == '__main__':
    main()