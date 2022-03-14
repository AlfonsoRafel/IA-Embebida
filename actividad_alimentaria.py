import argparse
from classification import run as class_run
from detection import run as det_run
import librosa
import numpy as np
from os.path import join
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import joblib

def generate_dataset(datadir):
    file = 'actividad_alimentaria.wav'
    labels = 'actividad_alimentaria.txt'
    
    total_events = []
    perc_bite = []
    perc_chew = []
    perc_chewbite = []
    part_classes = []
    df = pd.DataFrame({'events' : [], 'perc_bite' : [], 'perc_chew' : [], 'perc_chewbite' : [], 'class' : []})

    labels_dir = './data/data_actalim/actividad_alimentaria.txt'
    columns=['start', 'end','classes']
    predictions = pd.read_csv(labels_dir, sep='\t', header=None, names = columns)
    predictions.start= predictions.start.str.replace(',', '.')
    predictions.end= predictions.end.str.replace(',', '.')
    predictions.start = predictions.start.astype('float')
    predictions.end = predictions.end.astype('float')
    
    offset = 0
    duration_step = 3 * 60.0
    duration = librosa.get_duration(filename=join(datadir, file))
    steps = np.int(duration/duration_step)

    for i in range(0, steps - 1):
        part_class = 'noclass'
        print('Step: ', i)
        offset = i * duration_step
        data, sr = librosa.load(join(datadir, file), offset=offset, duration=duration_step)
        Ts = 1/sr
        N = data.shape[0]  
        #print('Offset: ', offset)
        t = np.arange(offset,Ts*N + i * duration_step,Ts)
        #print('Len data: ', len(data))
        #print('Sr: ', sr)
        #print('Start: ', t[0])
        #print('End: ', t[-1]) 
        for j in range(0, len(predictions.classes)):
            if (t[0]>= predictions.start[j]) & (t[0]< predictions.end[j]):
                part_class = predictions.classes[j]

        event_list, data_filt = det_run(data, sr)
        #print('Event List: ', event_list)
        t_set = np.arange(0,Ts*N,Ts)
        
        output_list = class_run(data,event_list, sr, t_set)
        output_array = np.array(output_list)
        events = len(output_array)
        total_events.append(len(output_array))
        perc_bite.append(len(np.where(output_array == 0)[0])/events)
        perc_chew.append(len(np.where(output_array  == 1)[0])/events)
        perc_chewbite.append(len(np.where(output_array  == 2)[0])/events)
        part_classes.append(part_class)

        
    df['events']= total_events
    df['perc_bite']= perc_bite
    df['perc_chew']= perc_chew
    df['perc_chewbite']= perc_chewbite
    df['class']= part_classes

    df.to_csv(join('./predict/act_alimentaria', 'out_clas_alim.csv'), index=False) 
    print('File generated')


def run(df):
    model = joblib.load('./models/model_act.pkl')

    for i in range(0, len(df.events)):
        #data = data_raw.astype(np.float32)
        X = np.array(df[['events' , 'perc_bite', 'perc_chew' , 'perc_chewbite']])
        X = X.reshape(1, len(X))
        Y = model.predict(X)
        df['class'][i] = Y[0]
    return df
        

def train(datadir):
    df = pd.read_csv('./predict/act_alimentaria/out_clas_alim.csv', sep=',', header=0)
    df['class'].replace(['Rumination', 'Grazing', 'End', 'noclass'], [0, 1, 2, 3], inplace= True)
    df = df.drop(df[df['class'] == 3].index)
    df = df.drop(df[df['class'] == 2].index)
    X_train = df[['events' , 'perc_bite', 'perc_chew' , 'perc_chewbite']]
    y_train = df['class']
    # Genera una nueva particion desde el conjunto de train
    X_train_opt, X_optim, y_train_opt, y_optim = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.33,
                                                                random_state=42)

        
    # Se define el modelo
    model = make_pipeline(
                            StandardScaler(),
                            #DecisionTreeClassifier(max_depth=5, min_samples_leaf=1),
                            #MLPClassifier(random_state=42, max_iter=2000, early_stopping=False, n_iter_no_change=100)
                            RandomForestClassifier(n_estimators=1, random_state=42)
                        )

    # Se entrena el modelo
    model.fit(X_train_opt, y_train_opt)

    # Se prueba con la partición de optimización
    y_pred_optim = model.predict(X_optim)

    sc = balanced_accuracy_score(y_optim, y_pred_optim)
    print(f"{sc:.3f} balanced accuracy")
    joblib.dump(model, './models/model_act.pkl')


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'datadir',
        metavar='path',
        default='None',
        help='Data root to be processed')
    args = arg_parser.parse_args()
    
    train(args.datadir)


if __name__ == '__main__':
    main()