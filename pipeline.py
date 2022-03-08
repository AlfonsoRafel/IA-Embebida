import argparse
from classification import predict as class_pred
from detection import predict as det_pred

def pipeline(datadir):
    det_pred(datadir)
    expdir = r'./predict/event_detection'
    class_pred(expdir)
    



    
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