import torch.multiprocessing as mp
from model import density_estimator
from torch.utils.cpp_extension import load
import torch as th
from torch import optim
import torch.distributions as dist 
from time import strftime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])

# no longer need write the batch loop in c - we can do that later
def get_batch(data_flag, batch_size, count, load_data=False):
    if not load_data:
        data = th.stack([amortized_rs.f() for _ in range(batch_size)]) 
    if data_flag == 'R1' and not load_data:
        inR1 = data[0:batch_size,0].view([batch_size,1]).view([1,batch_size])
        outR1 = data[0:batch_size,1].view([batch_size,1]).view([1,batch_size])
        count = count + 1 # not actually need if load_data == True
        return inR1.to(device),outR1.to(device), count
    elif data_flag == 'R1' and load_data:
        inR1 = data[count:batch_size+count*batch_size,0]
        outR1 = data[count:batch_size+count*batch_size,1]
        count = count + 1
        return inR1.to(device),outR1.to(device), count
    elif data_flag == 'R2'and not load_data:
        inR2 = th.cat([data[0:batch_size,1], data[0:batch_size,2]], dim=0).view(1,batch_size*2)
        outR2 = data[0:batch_size,3].view([1,batch_size])
        count = count + 1
        return inR2.to(device),outR2.to(device), count
    elif data_flag == 'R2' and load_data:
        inR2 = th.stack([data[count:batch_size+count*batch_size,1], data[count:batch_size+count*batch_size,2]], dim=1)
        outR2 = data[count:batch_size+count*batch_size,1]
        count = count + 1
        return inR2.to(device), outR2.to(device), count

def center_data(data):
    mean = th.mean(data,dim =0)
    zeroed = data - mean*th.ones(data.shape)
    stddata  = th.std(data, dim=0)
    return zeroed / stddata
def train(model, optimizer, loss_fn,  N, data_flag, batch_size, rank, load_data=False):
    model.train()
    n_samples = batch_size*N
    if load_data == True:
        data = th.load('../data/all_batch_samples.pt')
        data = data[0:n_samples,:]
    optimizer.zero_grad()
    count  = 0
    _outLoss = 0

    if not os.path.exists('../plots/'):
        os.makedirs('../plots/')

    # The network needs to learn z1 -> z2 and z2,z3 -> z4
    for i in range(N):
            inData, outData, count =get_batch(data_flag, batch_size, count)
            mu, std = model(inData)
            proposal = dist.Normal(mu, std)
            # pred = proposal.rsample(sample_shape=[batch_size]).view(1,batch_size))
            optimizer.zero_grad()
            # plt.show()
            # _loss = loss_fn(outData, pred)
            _loss = -proposal.log_prob(outData)
            total_loss = _loss.sum() / len(inData)
            total_loss.backward()
            optimizer.step()

            _outLoss += total_loss.item()
            iterationLoss = total_loss.item()
            # if _outLoss == nan:
            #     break
            if i % 20 == 0:
                # plt.show()
                # ax= sns.distplot(pred[0, :].detach().numpy(), kde=True, color='r',label='pred')
                # ax = sns.distplot(outData[0, :].detach().numpy(), kde=True, color='b', label='ground truth')
                # ax.legend()
                # ax.set_title('Iteration_{}_Rejection_block_{}'.format(i,data_flag))
                # fig = ax.get_figure()
                # fname= '../plots/{}_compare_pred_vs_gt_iteration_rejection_block_{}.png'.format(strftime("%M:%S"),data_flag)
                # fig.savefig(fname=fname)

                # plot_pred= sns.distplot(pred[0, :].detach().numpy(), kde=True, color='r')
                # plot_true =sns.distplot(outData[0, :].detach().numpy(), kde=True, color='b')
                # fnamePred = '../plot/' + 'predicted_iteration_{}_process_{}_rejectionBlock_{}'.format(i, rank, data_flag)
                # fnameTrue = '../plot/' + 'true_iteration_{}_process_{}_rejectionBlock_{}'.format(i, rank, data_flag)
                # plot_pred.savefig()
                avgLoss = _outLoss / (i+1)
                print("iteration {}: Average loss: {} Iteration loss: {}".format(i,avgLoss, iterationLoss))
            # print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, outloss))
    if not os.path.exists('../model/'):
        os.makedirs('../model/')
    fname = '../model/model_{}_rejectionBlock_{}_process_{}'.format(strftime("%Y-%m-%d_%H-%M"), data_flag, rank)
    th.save(model.state_dict(), fname)
    print(' Model is saved at : {}'.format(fname))

def test(model, test_iterations,model_name):
    model_name = '../model/' +model_name
    model.load_state_dict(th.load(model_name))
    model.eval()
    _outloss = 0
    count = 0
    with th.no_grad():
        for i in range(test_iterations):
            inData, outData, count = get_batch(data_flag, batch_size, count)
            proposal = dist.Normal(*model(inData))
            pred =proposal.rsample(sample_shape=[batch_size]).view(1,batch_size)
            plt.show()
            plt.title('Iteration : {}'.format(i))
            sns.distplot(pred, kde=True, color='r')
            sns.distplot(outData, kde=True, color='b')

            # learns a \hat{y} for the whole batch and then generates n_batch_size samples to predict the output.
            _loss = loss_fn(outData, pred)
            _outloss += _loss.item()
            if i % 100 == 0:
                print('{} Iteration , Test avg loss: {}\n'.format(i+1,_outloss/(i+1)))
        print('Test loss: {}\n'.format(_outloss/test_iterations))

def objective(zlearn, *args, **kwargs):
    ''' This has to be representative of the objective, equation 2'''
    return 0

if __name__ == '__main__':
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device("cpu")
    lr = 0.00001
    momentum= 0.60
    load_data=False
    batch_size = 2**10
    data_flag= 'R'
    outputSize = 1
    # outputSize = 128 # for R1 and R2
    if data_flag =='R2':
        inputSize = batch_size*2
    if data_flag == 'R1':
        inputSize = batch_size
    model = density_estimator(inputSize, outputSize)
    num_processes = mp.cpu_count() - 2
    N =  1000
    trainOn = True
    testOn = False
    # loss_fn = th.nn.CosineEmbeddingLoss()
    loss_fn = th.nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.01)
    train(model, optimizer, loss_fn, N, data_flag, batch_size, rank=0, load_data=False)
    # NOTE: this is required for the ``fork`` method to work
    # if trainOn:
    #     model.share_memory()
    #     processes = []
    #     for rank in range(num_processes):
    #         p = mp.Process(target=train, args=(model,optimizer,loss_fn, N,data_flag, batch_size, rank))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    # testOn = False
    if testOn:
        model_name = 'model_2019-08-12_09-35_rejectionBlock_R2_process_8'
        n_test = 1000
        test(model, n_test, model_name)
