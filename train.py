from    Instacne import Instance



def train():
    hyper, model, LossFn, optimizer, train_loader, test_loader = Instance()
    epoch = 0
    while epoch < hyper['ep']:
        epoch += 1
        for x, y in train_loader:
            x, y = x.to(hyper['dv']), y.to(hyper['dv'])


            pred = model(x)

