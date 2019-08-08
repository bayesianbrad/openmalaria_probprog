import torch.multiprocessing as mp
from model import density_estimator
from torch.utils.cpp_extension import load
import torch as th
from torch import optim
import torch.distributions as dist 
from time import strftime
import os
import time

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])

# no longer need write the batch loop in c - we can do that later
def get_batch(data_flag, batch_size, count, load_data=False):
    if not load_data:
        data = th.stack([amortized_rs.f() for _ in range(batch_size)]) 
    if data_flag == 'R1' and not load_data:
        inR1 = data[0:batch_size,0]
        outR1 = data[0:batch_size,1]
        count = count + 1 # not actually need if load_data == True
        return inR1.to(device),outR1.to(device), count
    elif data_flag == 'R1' and load_data:
        inR1 = data[count:batch_size+count*batch_size,0]
        outR1 = data[count:batch_size+count*batch_size,1]
        count = count + 1
        return inR1.to(device),outR1.to(device), count
    elif data_flag == 'R2'and not load_data:
        inR2 = th.stack([data[0:batch_size,1], data[0:batch_size,2]], dim=1)
        outR2 = data[0:batch_size,3]
        count = count + 1
        return inR2.to(device),outR2.to(device), count
    elif data_flag == 'R2' and load_data:
        inR2 = th.stack([data[count:batch_size+count*batch_size,1], data[count:batch_size+count*batch_size,2]], dim=1)
        outR2 = data[count:batch_size+count*batch_size,1]
        count = count + 1
        return inR2.to(device), outR2.to(device), count


    
def train(model, optimizer, loss_fn,  N, data_flag, batch_size, rank, load_data=False):
    model.train()
    n_samples = batch_size*N
    # data = th.stack([amortized_rs.f() for _ in range(batch_size)]) # hmm, shouldn't this be the number of training samples, rather than batch_size?
    # data = th.stack([amortized_rs.f() for _ in range(n_samples)]) # actually this loop needs to create multiple runs of the simulator of each z1, z2,z3 with
    if load_data == True:
        data = th.load('../data/all_batch_samples.pt')
        data = data[0:n_samples,:]
    optimizer.zero_grad()
    count  = 0
    _outLoss = 0

    # The network needs to learn z1 -> z2 and z2,z3 -> z4
    for i in range(N):
            inData, outData, count =get_batch(data_flag, batch_size, count)
            proposal = dist.Normal(*model(inData))
            pred = proposal.rsample()

            optimizer.zero_grad()
            _loss = loss_fn(outData, pred)
            _loss.backward()
            optimizer.step()

            _outLoss += _loss.item()

            if i % 10 == 0:
                avgLoss = _outLoss / (i+1)
                print("iteration {}: Average loss: {:6.3f}".format(i,avgLoss))
            # print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, outloss))
    if not os.path.exists('../model/'):
        os.makedirs('../model/')
    fname = '../model/model_{}_rejectionBlock_{}_process_{}'.format(strftime("%Y-%m-%d_%H-%M"), data_flag, rank)
    th.save(model.state_dict(), fname)
    print(' Model is saved at : {}'.format(fname))

def test(test_iterations,model_name):
    model = th.load(model_name)
    model.eval()
    _outloss = 0
    with th.no_grad():
        for i in range(test_iterations):
            inData, outData, count = get_batch(data_flag, batch_size, count)
            proposal = dist.Normal(*model(inData))
            pred = proposal.rsample()
            _loss = loss_fn(outData, pred)
            _outloss += _loss.item()
        print('Test loss: {:.4f}\n'.format(_outloss.item()/test_iterations))

def objective(zlearn, *args, **kwargs):
    ''' This has to be representative of the objective, equation 2'''

if __name__ == '__main__':
    loss_fn = th.nn.MSELoss()
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device("cpu")
    lr = 4e-5
    load_data=False
    # inputs = DataLoader(RejectionDataset(split='train', l_data=128, train_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True, norma num_workers=cpu_count-2)
    # outputs = DataLoader(RejectionDataset(split='test', l_data=128, test_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True,num_workers=cpu_count-2)
    batch_size = 128
    inputSize = batch_size
    data_flag= 'R1'
    outputSize = 1 # for R1
    model = density_estimator(inputSize, outputSize)
    num_processes = mp.cpu_count() - 1
    N = 100
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,optimizer,loss_fn, N,data_flag, batch_size, rank, load_data))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()