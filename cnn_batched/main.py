from data_loader import DataLoader
from trainer import Trainer

if __name__ == '__main__':
    datasets = DataLoader('../data/data17_npy/').get_datasets()
    Trainer(datasets).evaluate_lenet()
