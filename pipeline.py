import argparse

from numpy import int32
from classification import predict as class_pred
from detection import predict as det_pred
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np

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