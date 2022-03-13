import os
from collections import namedtuple  
import numpy as np 
import pandas as pd 
import scipy.io.wavfile 
import scipy.signal 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import welch
from scipy.signal import peak_prominences





WaveData = namedtuple("WaveData", ["data", "sample_freq", "label"])

class WaveDataset():
    
    def __init__(self, data_folder, annotation_file):
        """Construye un dataset desde carpeta de wav y csv de labels."""
        # Carpeta donde estan los wav
        self.data_folder = data_folder 
        
        # Crea una lista de nombres y etiquetas a partir del csv
        if annotation_file is not None:
            ds = pd.read_csv(annotation_file)
            self.filenames = list(ds['filename'])
        
            if 'label' in ds.columns:
                # Si existe una columna de labels la toma del csv
                self.labels = ds['label'].values
            else:
                # Caso contrario, los labels son todos -1
                self.labels = -np.ones(len(self.filenames))   
        self.cache = {}
    
    #====================================================================
    def __len__(self):
        """Cantidad de patrones en el dataset."""
        return len(self.labels)
    
    #====================================================================
    def __getitem__(self, index):
        """Retorna la tupla (data, fs, label) en la posicion index."""
        if index in self.cache:
            data, fs, label = self.cache[index]
        else:
            # Lee el archivo wav y guarda (muestras, label) en cache
            fname = f"{self.filenames[index] :04d}.wav"
            fpath = os.path.join(self.data_folder, fname)
            
            fs, data = scipy.io.wavfile.read(fpath) # lectura de wavs con scipy
            data = data / np.iinfo(data.dtype).max  # Reescala entre -1 y 1
            
            # data, fs = librosa.load(fpath)  # lectura de wavs con librosa
            
            label = self.labels[index]
            self.cache[index] = (data, fs, label)
            
        return WaveData(data, fs, label)
    
def preprocessing(raw_signal, sample_freq):

    # Rectification 
    rectified_signal = np.abs(raw_signal)

    # Low-pass filtering -> envelope
    cutoff_freq = 5  # Hz
    sos_butter = scipy.signal.butter(2, cutoff_freq, 'lowpass', output="sos", fs=sample_freq)
    envelope = scipy.signal.sosfiltfilt(sos_butter, rectified_signal, axis=0)

    return envelope, rectified_signal

def extract_features(raw_signal, sample_freq, plot_features=False):
    
    envelope, rectified_signal = preprocessing(raw_signal, sample_freq)
     
    threshold = 0.0230  # umbral para duracion y signo de la pendiente de la envolvente
    above_thres = envelope >= threshold  # arreglo booleano para restringir zona de calculo
    
    # variables y señales auxiliares
    max_pos = np.argmax(envelope)  # posicion del maximo en la ventana
    
    # al multiplicar por above_thres hace cero por debajo del umbral (enmascara)
    envelope_slope_sign = np.sign(np.diff(envelope, append=0)) * above_thres
    duration_signal = np.ones(envelope.shape) * above_thres  

    # calcular el ancho max de los picos dentro del evento
    peaks, properties = find_peaks(envelope)
    widths = peak_widths(envelope, peaks)
    heights = peak_prominences(envelope, peaks)
   
    
    # Caracteristicas
    duration = np.sum(duration_signal) / sample_freq
    zero_crossing = np.count_nonzero(np.abs(np.diff(envelope_slope_sign)))
    amplitude = np.max(rectified_signal)
    ratio = np.trapz(envelope[:max_pos], axis=0) / np.trapz(envelope, axis=0)
    std = np.std(raw_signal)
    mean = np.mean(envelope)
    width = np.max(widths[0])/len(envelope)
    energy = np.sum(raw_signal*raw_signal)/100
    height = np.max(heights[0])

    features = [duration, zero_crossing, amplitude, ratio, std, width, height, energy]  
    
    # Grafica las caracteristicas
    if plot_features:
        
        t = np.arange(0, len(raw_signal)/sample_freq, 1/sample_freq)
        
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(24,12), sharex=True)
        axes[0].plot(t, raw_signal)
        axes[0].set_title("Señal sonora")
        axes[0].grid(True)
        
        axes[1].plot(t, envelope)
        axes[1].set_title("Envolvente")
        axes[1].plot([0, t.max()], [threshold, threshold], "--")
        axes[1].legend(["Envolvente", "Umbral"])
        axes[1].grid(True)
        
        axes[2].plot(t, envelope_slope_sign)
        axes[2].set_title("Signo de la pendiente de la envolvente")
        axes[2].grid(True)
        
        axes[3].plot(t, duration_signal)
        axes[3].set_title("Duración")
        axes[3].set_xlabel("Tiempo [s]")
        axes[3].grid(True)
        
        fig.tight_layout()
        plt.show()
        
        print(["Duración (s)", "Cruces por cero", "Amplitud máxima", "Simetria", "Desvio"])
        print(features)
    
    return features



class WaveDatasetFull():
    
    def __init__(self, data_folder, annotation_file):
        """Construye un dataset desde carpeta de wav y csv de labels."""
        # Carpeta donde estan los wav
        self.data_folder = data_folder 
        
        # Crea una lista de nombres y etiquetas a partir del csv
        if annotation_file is not None:
            ds = pd.read_csv(annotation_file)
            self.filenames = list(ds['filename'])
        
            if 'label' in ds.columns:
                # Si existe una columna de labels la toma del csv
                self.labels = ds['label'].values
                self.height = ds['height'].values
                self.pasture = ds['pasture'].values
            else:
                # Caso contrario, los labels son todos -1
                self.labels = -np.ones(len(self.filenames))   
        self.cache = {}
    
    #====================================================================
    def __len__(self):
        """Cantidad de patrones en el dataset."""
        return len(self.labels)
    
    #====================================================================
    def __getitem__(self, index):
        """Retorna la tupla (data, fs, label) en la posicion index."""
        if index in self.cache:
            data, fs, label, pasture, height = self.cache[index]
        else:
            # Lee el archivo wav y guarda (muestras, label) en cache
            fname = f"{self.filenames[index] :04d}.wav"
            fpath = os.path.join(self.data_folder, fname)
            
            fs, data = scipy.io.wavfile.read(fpath) # lectura de wavs con scipy
            data = data / np.iinfo(data.dtype).max  # Reescala entre -1 y 1
            
            # data, fs = librosa.load(fpath)  # lectura de wavs con librosa
            
            label = self.labels[index]
            self.cache[index] = (data, fs, label)
            
        return WaveData(data, fs, label, pasture, height)