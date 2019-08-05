
import torch
from model import density_estimator
from time import strftime
import os 
import sys
from torch import optim
import torch.distributions as dist
from torch.multiprocessing import cpu_count



inputs = DataLoader(RejectionDataset(split='train', l_data=128, train_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True, norma num_workers=cpu_count-2)
outputs = DataLoader(RejectionDataset(split='test', l_data=128, test_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True,num_workers=cpu_count-2)

epochs = 10
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)

model.to(device)



    model.eval()
    b_loss = 0
    test_iterations = len(test_loader)
    with torch.no_grad():
        for i, (a,b) in enumerate(test_loader):
            a, b = a.to(device), b.to(device)
            proposal = dist.Normal(*model(a))
            pred = proposal.rsample()
            _loss = loss_fn(b, pred)
            b_loss += loss.item()
        print('Test loss: {:.4f}\n'.format(b_loss.item()/test_iterations))

# inputDimR1 = 1        # takes variable 'x' from R1 
# outputDimR1 = 1       # takes variable 'r(x)' from R1
# inputDimR1 = 2        # takes variable 'x' from R2 
# outputDimR1 = 1       # takes variable 'r(x)' from R2
# learningRate = 0.01 
# epochs = 100

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DensityEstimator():
    """
    Class to estimate the density. The class takes a pytorch optmizer
    """

    def __init__(self, device, optmizer):
        self.device = device
        self.optmizer = optmizer
        self.steps = 0

    def train(self, epochs, data):
        """
        Method to train the density estimator. 

        :param: epochs :type: int :descrp: Number of epochs to train for.
        :param: data :type: torch.Tensor  :descrp: The data to pass through the network. 

        :return: Trained model
        """

        for epoch in range(epochs):
            model.train()
            train_iterations = len(train_loader)
            for i, (a, b) in enumerate(train_loader):
                a, b = a.to(device), b.to(device)
                proposal = dist.Normal(*model(a))
                pred = proposal.rsample()

                optimizer.zero_grad()
                _loss = loss_fn(b, pred)
                _loss.backward()
                optimizer.step()

                b_loss += loss.item()
                if args.print_freq > 0 and i % args.print_freq == 0:
                    print("iteration {:04d}/{:d}: loss: {:6.3f}".format(i, iterations,
                                                                        loss.item() / args.batch_size))
                print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, ...))

