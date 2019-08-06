import utils
from rejection_samplers import frepeat


<<<<<<< HEAD
PATH = '/home/bradley/projects/openmalaria_probprog/code/amortized_rs/batch_samples/'

def main():
    data=utils.load_samples(PATH=PATH)
=======
PATH = '/home/bradley/projects/openmalaria_probprog/code/amortized_rs/run_2/samples_with_inputs/'
learning = False

def main():
    data=utils.load_samples(PATH=PATH, learning=learning)
>>>>>>> e2910a3683b657af5fa91292c678419ec3b427c7
    batch_size=128
    totalSamples=len(data)
    simulator=frepeat

    utils.create_dataset(data=data,batch_size=batch_size, totalSamples=totalSamples, simulator=simulator)

if __name__ == "__main__":
    main()

