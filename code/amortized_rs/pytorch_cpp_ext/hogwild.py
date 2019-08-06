import torch.multiprocessing as mp
from model import MyModel
from torch.utils.cpp_extension import load
import torch as th
from torch import optim
from torch.distributions import Normal
from utils import RejectionDataset
from torch.utils.data import DataLoader

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])
loss_fn = th.nn.MSELoss()
device = th.device("cuda" if th.cuda.is_available() else "cpu")
inputs = DataLoader(RejectionDataset(split='train', l_data=128, train_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True, norma num_workers=cpu_count-2)
outputs = DataLoader(RejectionDataset(split='test', l_data=128, test_percentage=0.8,fname_test, fname_train, InIndx, OutIndx), batch_size=128, shuffle=True,num_workers=cpu_count-2)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], amsgrad=True)

def train(model, max_it, data_flag):
    # Construct data_loader, optimizer, etc.
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for it in range(max_it):
        model.train()
        batch_size = 128
        N = 50000
        n_samples = batch_size*N
        # data = th.stack([amortized_rs.f() for _ in range(batch_size)]) # hmm, shouldn't this be the number of training samples, rather than batch_size?
        # data = th.stack([amortized_rs.f() for _ in range(n_samples)]) # actually this loop needs to create multiple runs of the simulator of each z1, z2,z3 with
        data = torch.load('/data/all_batch_samples.pt')
        data = data[0:n_samples,:]
        optimizer.zero_grad()
        # TODO: Extract data, labels and define loss_fn etc!
        count  = 0
        def get_batch(data_flag, batch_size, count):
            if data_flag == 'R1':
                inR1 = data[count:batch_size+count*batch_size,0]
                outR1 = data[count:batch_size+count*batch_size,1]
                count = count + 1
                return inR1.to(device),outR1.to(device), count
            elif data_flag == 'R2':
                inR2 = th.stack([data[count:batch_size+count*batch_size,1], data[count:batch_size+count*batch_size],2], dim=1))
                outR2 = data[count:batch_size+count*batch_size,3]
                count = count + 1
            return inR2.to(device),outR2.to(device), count

        # The network needs to learn z1 -> z2 and z2,z3 -> z4
        for i in range(N):
                inData, outData, count =get_batch(data_flag, batch_size, count)
                proposal = dist.Normal(*model(inData))
                pred = proposal.rsample()

                optimizer.zero_grad()
                _loss = loss_fn(outData, pred)
                _loss.backward()
                optimizer.step()

                outLoss += loss.item()

                if args.print_freq > 0 and i % args.print_freq == 0:
                    print("iteration {:04d}/{:d}: loss: {:6.3f}".format(i, iterations,
                                                                        loss.item() / args.batch_size))
                print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, ...))

def test():
    model.eval()
    out_loss = 0
    test_iterations = len(test_loader)
    with torch.no_grad():
        for i, (inData,outData) in enumerate(test_loader):
            inData, outData = inData.to(device), outData.to(device)
            proposal = dist.Normal(*model(inData))
            pred = proposal.rsample()
            _loss = loss_fn(outData, pred)
            _outloss += loss.item()
        print('Test loss: {:.4f}\n'.format(_outloss.item()/test_iterations))

if __name__ == '__main__':
    num_processes = 4
    max_it = 10000
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,max_it))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()