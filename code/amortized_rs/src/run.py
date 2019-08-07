import utils
from rejection_samplers import frepeat


PATH = '/home/bradley/projects/openmalaria_probprog/code/amortized_rs/batch_samples/'

def main():
    data=utils.load_samples(PATH=PATH)
    batch_size=128
    totalSamples=len(data)
    simulator=frepeat

    utils.create_dataset(data=data,batch_size=batch_size, totalSamples=totalSamples, simulator=simulator)

if __name__ == "__main__":
    main()

