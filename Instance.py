from    Datasets.CoCoSet import CoCo
from    Models.HRNet import HRNet
from    torch import nn
from    torch.utils.data import DataLoader

import torch


def Instance():
    hyper = {'lr': 1e-4, 'bz': 32, 'ep': 500,
             'dv': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

    model = HRNet().to(hyper['dv'])
    LossFn = nn.MSELoss().to(hyper['dv'])
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper['lr'], weight_decay=0.4, momentum=0.7)

    train_db = CoCo('train')
    test_db = CoCo('test')

    train_loader = DataLoader(train_db, batch_size=hyper['bz'],
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_db, batch_size=hyper['bz'],
                             shuffle=True, num_workers=2)
    return hyper, model, LossFn, optimizer, train_loader, test_loader