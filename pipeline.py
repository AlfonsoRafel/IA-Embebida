import argparse

from numpy import int32
from classification import predict as class_pred
from detection import predict as det_pred
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import librosa
from os.path import join

from classification import run as class_run
from detection import run as det_run
from actividad_alimentaria import run as act_run


def pipeline(datadir):
    det_pred(datadir)
    expdir = './predict/event_detection'
    class_pred(expdir)
    validate(datadir)
    
def validate(datadir):
    # ground thruth
    truth_dir = datadir.replace(".wav", ".txt")
    columns=['start','end','classes']
    thruth = pd.read_csv(truth_dir, sep='\t', header=None, names = columns)
    thruth.classes = thruth.classes.replace(['bite', 'chew', 'chewbite'], [0, 1, 2])

    # prediction
    pred_dir = './predict/classification/out.csv'
    columns=['name','classes']
    predictions = pd.read_csv(pred_dir, sep=',', header=0, names = columns)
    predictions.name = predictions.name.str.replace('.wav', '')
    predictions.name = predictions.name.astype('int32')
    predictions.sort_values(by=['name'], inplace=True)
    print(predictions)


    len_min = np.minimum([len(thruth.classes)], [len(predictions.classes)])

    sc = balanced_accuracy_score(thruth.classes[0:len_min[0]], predictions.classes[0:len_min[0]])
    print(f"{sc:.3f} balanced accuracy")

def pipeline_full(datadir, step):
    
    total_events = []
    perc_bite = []
    perc_chew = []
    perc_chewbite = []
    part_classes = []
    df = pd.DataFrame({'events' : [], 'perc_bite' : [], 'perc_chew' : [], 'perc_chewbite' : [], 'class' : []})

    
    offset = 0
    duration_step = step * 60.0
    duration = librosa.get_duration(filename=join(datadir))
    steps = np.int(duration/duration_step)

    for i in range(0, steps - 1):
        print('Step: ', i)
        offset = i * duration_step
        data, sr = librosa.load(join(datadir), offset=offset, duration=duration_step)
        Ts = 1/sr
        N = data.shape[0]  
        t = np.arange(offset,Ts*N + i * duration_step,Ts) 

        event_list, data_filt = det_run(data, sr)
        t_set = np.arange(0,Ts*N,Ts)
        
        output_list = class_run(data,event_list, sr, t_set)
        output_array = np.array(output_list)
        events = len(output_array)
        total_events.append(len(output_array))
        perc_bite.append(len(np.where(output_array == 0)[0])/events)
        perc_chew.append(len(np.where(output_array  == 1)[0])/events)
        perc_chewbite.append(len(np.where(output_array  == 2)[0])/events)
    
    df['events']= total_events
    df['perc_bite']= perc_bite
    df['perc_chew']= perc_chew
    df['perc_chewbite']= perc_chewbite
    
    df_new = act_run(df)
    start = []
    end = []
    label = []
    for i in df_new.index:
        start.append(i * 3 * 60.0)
        end.append(i * 3 * 60.0 + 3*60.0)
        if df_new['class'][i] == 0:
            label.append('Rumination')
        else:
            label.append('Grazing')
    df = pd.DataFrame({'start' : [], 'end' : [], 'class' : []})
    df_new['start'] = start
    df_new['end'] = end
    df_new['class'] = label
    df_new.sort_values(by=['start'], inplace=True)
    df_new.to_csv('prediction.csv', index=False) 
    print('Predicción Realizada')

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'datadir',
        metavar='path',
        default='None',
        help='Data root to be processed')
    args = arg_parser.parse_args()
    pipeline(args.datadir)

if __name__ == '__main__':
    main()