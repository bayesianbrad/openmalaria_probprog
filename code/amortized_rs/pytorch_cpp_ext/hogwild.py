import torch.multiprocessing as mp
from model import MyModel
from torch.utils.cpp_extension import load
import torch as th
from torch import optim

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])

def train(model, max_it):
    # Construct data_loader, optimizer, etc.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for it in range(max_it):
        batch_size = 256
        data = th.stack([amortized_rs.f() for b in range(batch_size)])
        optimizer.zero_grad()
        # TODO: Extract data, labels and define loss_fn etc!
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

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