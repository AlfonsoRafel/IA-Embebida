import argparse
from classification import run as class_run
from detection import run as det_run
import librosa
import numpy as np
from os.path import join
import pandas as pd

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

def predict():
    return NotImplemented
        

def train(datadir):
    
    return NotImplemented


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'datadir',
        metavar='path',
        default='None',
        help='Data root to be processed')
    args = arg_parser.parse_args()
    
    generate_dataset(args.datadir)


if __name__ == '__main__':
    main()