from benchmarks.RPDBCS import experiment_finetunning
from vibdata.datahandler import RPDBCS_raw, MFPT_raw
from vibdata.datahandler.transforms.TransformDataset import PickledDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # D = RPDBCS_raw('datasets', frequency_domain=True)
    # D = MFPT_raw('datasets')
    # D = PickledDataset('/tmp/sigdata_cache/rpdbcs')
    # dl = DataLoader(D, batch_size=32, shuffle=True, drop_last=True)
    # for d in dl:
    #     print(d)
    #     break
    R = experiment_finetunning.run_experiment('datasets')
    print(R)
    R.to_csv('results.csv', index=False)
