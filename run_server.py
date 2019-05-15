#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import traceback
import argparse
import sys
import os
from glob import glob
import pprint
import uuid
import time
import subprocess
import signal
import pypdt
import numpy as np
import torch
import pyprob
from pyprob import RemoteModel, AddressDictionary, PriorInflation, InferenceEngine, InferenceNetwork, ObserveEmbedding, LearningRateScheduler
import pyprob.diagnostics
from pyprob.util import get_time_stamp, to_tensor, to_numpy, days_hours_mins_secs_str, progress_bar_init, progress_bar_update, progress_bar_end
from pyprob.nn import InferenceNetworkLSTM, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, OfflineDataset
import csv
import random
from collections import defaultdict, OrderedDict
from spython.main import Client

parser = argparse.ArgumentParser(description='etalumis sherpa tau decay experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--device', help='Set the compute device (cpu, cuda:0, cuda:1, etc.)', default='cpu', type=str)
        parser.add_argument('--seed', help='Random number seed', default=None, type=int)
        parser.add_argument('--model_executable', help='Model executable to start as a background process', default='/examples/openMalaria')
        parser.add_argument('--model_address', help='PPX protocol address', default=None, type=str)
        # parser.add_argument('--mode', help='', choices=['create_truths', 'prior', 'infer', 'plot', 'plot_diagnostics', 'plot_diagnostics_network', 'plot_diagnostics_addresses', 'save_numpy', 'save_dataset', 'train'], nargs='?', default='', type=str)
        parser.add_argument('--mode', '-m', help='Experiment mode', choices=['ground_truth', 'ground_truth_from_dataset', 'prior', 'posterior', 'plot', 'plot_log_prob', 'plot_autocorrelation', 'plot_gelman_rubin', 'plot_addresses', 'plot_traces', 'plot_graph', 'plot_network', 'diag_address_dict', 'diag_mmd', 'diag_dataset', 'combine', 'save_numpy', 'save_dataset', 'sort_dataset', 'train'], nargs='?', default=None, type=str)
        parser.add_argument('--num_traces', '-n', help='Number of traces for various tasks', default=None, type=int)
        parser.add_argument('--num_traces_end', help='Target number of traces for IC network training', default=1e12, type=int)
        parser.add_argument('--num_traces_per_file', help='Number of traces per file for offline dataset generation', default=None, type=int)
        parser.add_argument('--num_files', help='Number of files for various tasks', default=None, type=int)
        parser.add_argument('--begin_file_index', help='Index of first file in multi-node trace sorting. If not given, taken as 0.', default=None, type=int)
        parser.add_argument('--end_file_index', help='Index (not inclusive) of last file in trace sorting. If not give, taken as the last index in the dataset.', default=None, type=int)
        parser.add_argument('--distributed_backend', help='Backend to use with distributed training', choices=['mpi'], nargs='?', default=None, type=str)
        parser.add_argument('--distributed_num_buckets', help='Number of buckets in distributed training', default=None, type=int)
        parser.add_argument('--input_file', '-i', help='Input file(s) for various tasks', default=[], action='append', type=str)
        parser.add_argument('--output_file', '-o', help='Output file for various tasks', default=None, type=str)
        parser.add_argument('--input_dir', help='Input directory for various tasks', default=None, type=str)
        parser.add_argument('--output_dir', help='Output directory for various tasks', default=None, type=str)
        parser.add_argument('--infer_engine', '-e', help='RMH: MCMC with Random-walk Metropolis Hastings, IS: Importance sampling, IC: Inference compilation', choices=['RMH', 'IS', 'IC'], nargs='?', type=str)
        parser.add_argument('--infer_init', help='Initialization of RMH inference (prior: start from a random trace sampled from prior, ground_truth: start from the ground truth trace)', choices=['prior', 'ground_truth'], nargs='?', default='ground_truth', type=str)
        parser.add_argument('--likelihood_importance', help='Importance factor of the observation likelihood in inference', default=1., type=float)
        parser.add_argument('--network_type', help='Type of inference neural network to train for IC inference', default=None, choices=['feedforward', 'lstm'], nargs='?', type=str)
        parser.add_argument('--optimizer', help='Type of optimizer to train NNs for IC inference', choices=['adam', 'sgd', 'adam_larc', 'sgd_larc'], nargs='?', default='adam', type=str)
        parser.add_argument('--learning_rate_init', help='IC training learning rate (initial)', default=0.0001, type=float)
        parser.add_argument('--learning_rate_end', help='IC training learning rate (end)', default=1e-6, type=float)
        parser.add_argument('--learning_rate_scheduler', help='Learning rate sceduler (step, multistep)', choices=['poly1', 'poly2'], nargs='?', default=None, type=str)
        parser.add_argument('--min_index', help='Starting index of distributions for various tasks', default=None, type=int)
        parser.add_argument('--max_index', help='End index of distributions for various tasks', default=None, type=int)
        parser.add_argument('--prior_inflation', help='Use prior inflation', action='store_true')
        parser.add_argument('--channel', help='Generate a specific decay channel event in mode:ground_truth', default=None, type=int)
        parser.add_argument('--ground_truth', '-g', help='Ground truth file for various tasks', default=None, type=str)
        parser.add_argument('--observation', help='Observation to extract from ground truth trace', choices=['value', 'mean'], nargs='?', default='mean', type=str)
        parser.add_argument('--address_dict', '-a', help='Address dictionary', default=None, type=str)
        parser.add_argument('--dataset_dir', help='Dataset directory for offline training', default=None, type=str)
        parser.add_argument('--dataset_valid_dir', help='Dataset directory for offline validation', default=None, type=str)
        parser.add_argument('--network_dir', help='Dictionary to keep inference neural network snapshots', default=None, type=str)
        parser.add_argument('--log_file_name', help='Log file for training', default=None, type=str)
        parser.add_argument('--use_address_base', help='Use base addresses (ignore iteration counters) in diagnostics plots', action='store_true')
        parser.add_argument('--normalize_graph_weights', help='Normalize edge weights per source node in diagnostics graph plots', action='store_true')
        parser.add_argument('--batch_size', help='Minibatch size for IC training', default=None, type=int)
        parser.add_argument('--valid_size', help='Validation size for IC training (online training only)', default=None, type=int)
        parser.add_argument('--valid_every', help='Number of traces between validations for IC training', default=None, type=int)
        parser.add_argument('--save_every', help='Interval for saving neural network to disk during IC training', default=None, type=int)
        parser.add_argument('--dataloader_offline_num_workers', help='Number of worker threads for data loading in offline training mode', default=0, type=int)
        parser.add_argument('--n_most_frequent', help='Number of most frequent trace types to use for graph construction (mode: plot_graph)', default=None, type=int)
        parser.add_argument('--thinning_steps', help='Number used for online thinning. Specifies how many samples to skip before saving a sample used for inference (mode: posterior, infer_engine: RHM)', default=None, type=int)
        parser.add_argument('--lstm_dim', help='LSTM number of hidden units', default=512, type=int)
        parser.add_argument('--lstm_depth', help='LSTM number of stacked layers', default=1, type=int)
        parser.add_argument('--proposal_mixture_components', help='Number of mixture components to use in proposal distributions', default=10, type=int)
        parser.add_argument('--mmd_num_null_samples', help='Number of samples from MMD null distribution estimate (by permutation)', default=1000, type=int)
        parser.add_argument('--host_container_to_build', help='Host Container. Will take in a docker container tag as a string and will build to a singularity container, the container that runs the executable. ', default=None, type=str)
        parser.add_argument('--host_container', help='Host Container location. Will take in a Singularity .simg container, the container that runs the executable. ',default=None, type=str)
        parser.add_argument('--client_container_to_build', help='Client Container. Will take in a docker container tag as a string and will build to a singularity container. ', default=None, type=str)
        parser.add_argument('--client_container', help='Client Container. Will take in a Singularity .simg container as a string and run client operatorations inside. ', default=None, type=str)
        opt = parser.parse_args()

        if opt.mode is None:
            parser.print_help()
            quit()

        print('Run Simulator\n')
        print('Mode:\n{}\n'.format(opt.mode))

        print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))

        print('Config:')
        pprint.pprint(vars(opt), depth=2, width=50)

# parser = argparse.ArgumentParser(description='Parse OM server arguments')
# parser.add_option('-p', '--pop_size', type=int)
# parser.add_option()
# args = parser.parse_args()

# population_size = args.pop_size
def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        try:
            os.makedirs(dir)
        except Exception as e:
            print(e)
            print('Could not create path, potentially be created by another rank in multinode: {}'.format(path))

# Inside the host container we have  is everything to run the execuatble
# inside the client contaijer we have the pyprob environment and nay other local python packaged that are wanted for  postprocessing
def main():
    # print(' Starting singularatiy container for the host {}'.format(opt.host_container))
    # if opt.host_container is None:
    #     __host = Client.load(opt.host_container)
    # elif opt.host_container_to_build:
    #     __host = Client.load(opt.host__to_build_container)
    # # add something here to create an arbrtiary number of host containers and run them all over separate addresses
    # run = __host.execute([{'./'+opt.model_executable + opt.model_address}])

    print('Starting model in the background: {} {}'.format(opt.model_executable, opt.model_address))
    model_process = subprocess.Popen('{} {} > /dev/null &'.format(opt.model_executable, opt.model_address), shell=True, preexec_fn=os.setsid)
    print('Started model process: {}'.format(model_process.pid))
    model = RemoteModel(opt.model_address, address_dict_file_name=opt.address_dict)
    print('Saving prior distribution with {} traces to: {}'.format(opt.num_traces, opt.output_file))
    create_path(opt.output_file)
    prior_dist = model.prior_traces(num_traces=opt.num_traces, file_name=opt.output_file)


# import subprocess
# def main():
#     print('Starting model in the background: {} {}'.format('/openMalaria', 'ipc://@openpp'))
#     model_process = subprocess.Popen('{} {} > /dev/null &'.format('/openMalaria', 'ipc://@openpp'), shell=True, preexec_fn=os.setsid)
#     print('Started model process: {}'.format(model_process.pid))
#     model = RemoteModel('ipc://@openpp', address_dict_file_name='address_dict.txt')
#     print('Saving prior distribution with {} traces to: {}'.format(1, 'trace_output.txt'))
#     create_path('trace_output.txt')
#     prior_dist = model.prior_traces(num_traces=1, file_name='trace_output.txt')
if __name__ == "__main__":
    time_start = time.time()
    main()
    print('\nTotal duration: {}'.format(days_hours_mins_secs_str(time.time() - time_start)))
    sys.exit(0)
