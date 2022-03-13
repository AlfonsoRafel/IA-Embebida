from ast import Pass
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from os.path import join, isfile
from os import listdir
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks
import scipy
import json
import statistics

def load_data(datadir, filename):
    fs, data_raw = wavfile.read(join('r',datadir, filename), False)
    Ts = 1/fs
    N = data_raw.shape[0]  
    t = np.arange(0,Ts*N,Ts)  
    columns=['start','end','class']
    labels =filename.replace('.wav', '.txt')
    data_labels = pd.read_csv(join('r',datadir, labels), sep='\t', header=None, names = columns)
    data = data_raw.astype(np.float32)
    return data, data_labels, fs, t, N

def normalization(data):
    # normalization 
    data = (data - data.min())
    data = data / (data.max() - data.min())
    data = 2*(data - data.mean())
    return data

def filter_4000hz(data, fs):
    f_nyq = fs/2
    order, wn = scipy.signal.buttord(wp=2000,       # Banda de paso [Hz]
                                 ws=2500,       # Banda de rechazo [Hz]
                                 gpass=3,       # Atenuación de 3 dB en banda de paso
                                 gstop=40,      # Atenuación de 60 dB en banda de stop
                                 analog=False,  # Digital
                                 fs=fs)         # [Hz]
    b, a = scipy.signal.butter(order,
                           wn,
                           btype='lowpass',
                           analog=False,
                           fs=fs)
    data_filt_2K = scipy.signal.lfilter(b, a, data)
    return data_filt_2K

def filter_event(data, fs):
    # Filter for event detection

    order, wn = scipy.signal.buttord(wp=5,          # Banda de paso [Hz]
                                 ws=20,         # Banda de rechazo [Hz]
                                 gpass=3,       # Atenuación de 3 dB en banda de paso
                                 gstop=40,      # Atenuación de 40 dB en banda de stop
                                 analog=False,  # Digital
                                 fs=fs)         # [Hz]
    b, a = scipy.signal.butter(order,
                           wn,
                           btype='lowpass',
                           analog=False,
                           fs=fs)

    X = np.float32(data)**2
    data_filt_5 = scipy.signal.lfilter(b, a, X)
    data_filt = 6 * data_filt_5
    return data_filt

def get_events(data, mean, std, t, Ts, N):
    peaks, properties = find_peaks(data, distance = 15000, prominence=0.01, width=2000)
    event_list = []
    peaks_time = peaks/len(t) * (Ts*N)
    
    for peak in peaks_time:
        start = peak - mean/2 - std
        end = peak + mean/2 - std
        event_list.append((start, end))
    return event_list

def cal_IoU(event_list, data_labels, t):
    # metric
    y_hat = t.copy()
    y = t.copy()
    for idx in range(len(data_labels)):
            start = data_labels['start'].iloc[idx] 
            end =  data_labels['end'].iloc[idx]
            y[np.where((y<end) & (y>=start))] = 1
    y[np.where(y!= 1)] = 0


    for idx in range(len(event_list)):
            start = event_list[idx][0]
            end =  event_list[idx][1]
            y_hat[np.where((y_hat<end) & (y_hat>=start))] = 1
    y_hat[np.where(y_hat!= 1)] = 0
    
    intersection = np.minimum(y, y_hat)
    union = y + y_hat
    union[np.where(union > 1)] = 1

    inter = np.sum(intersection * t)
    union = np.sum(union * t)

    ratio = np.round(inter/union, 2) * 100
    print(f"{ratio:.3f} IoU")
  
def save_event(event_list, data, fs, t):
    savedir = './predict/event_detection'
    for event in range(0, len(event_list)):
        start = event_list[event][0]
        end =  event_list[event][1]
        split_data = data[np.where((t<end) & (t>=start))] 
        wavfile.write(join(savedir, str(event) + '.wav'), fs, split_data)

        
  
  
  
def predict(filedir):
    fs, data_raw = wavfile.read(join(filedir), False)
    Ts = 1/fs
    N = data_raw.shape[0]  
    t = np.arange(0,Ts*N,Ts)
    data = data_raw.astype(np.float32)
    data = normalization(data)
    data_filt_2K = filter_4000hz(data, fs)
    data_filt = filter_event(data_filt_2K, fs)
    
    # load json
    json_file = './models/detection.json'
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
        except:
            Pass
    mean = statistics.mean(config_dict['Mean'])
    std = statistics.stdev(config_dict['Std'])     
    event_list = get_events(data_filt, mean, std, t , Ts, N) 
    save_event(event_list, data_raw, fs, t)
    print('Events detected and saved')
    
       
def train(datadir):
    exp_path = './models'
    expdir = exp_path
    
    filenames = [f for f in listdir(datadir) if (isfile(join(datadir, f)) and (f.endswith('.wav')))]
    # mean and std
    mean_list = []
    std_list = []
    
    # load data
    for filename in filenames:
        data, data_labels, fs, t, N = load_data(datadir, filename)
        Ts = 1/fs
        data = normalization(data)
        data_filt_2K = filter_4000hz(data, fs)
        data_filt = filter_event(data_filt_2K, fs)
        
        mean_list.append(np.mean(data_labels.end - data_labels.start))
        std_list.append(np.std(data_labels.end - data_labels.start))
        
        event_list = get_events(data_filt, mean_list[-1], std_list[-1], t , Ts, N)
        cal_IoU(event_list, data_labels, t)
        
    # save json
    out_dict = {}
    out_dict['Mean'] = mean_list
    out_dict['Std'] = std_list
    outdir = join(expdir, "detection.json")
    out_json = open(outdir, "w")
    json.dump(out_dict, out_json)
        




    
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