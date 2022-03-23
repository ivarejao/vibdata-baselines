from benchmarks.RPDBCS import experiment_finetunning2
# from vibdata.datahandler import RPDBCS_raw, MFPT_raw, XJTU_raw
from vibdata.datahandler.transforms.TransformDataset import PickledDataset

if __name__ == '__main__':
    from sys import argv
    # D = XJTU_raw('datasets', download=True)
    R = experiment_finetunning2.run_experiment('datasets')
    print(R)
    if(len(argv) > 1):
        foutname = argv[1]
    else:
        foutname = 'results.csv'
    R.to_csv(foutname, index=False)
