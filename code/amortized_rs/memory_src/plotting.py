#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  10:12
Date created:  04/09/2019

License: MIT
'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:14
Date created:  09/01/2018

License: MIT
'''

import torch
import numpy as np
import sys
import os
import copy
import seaborn as sns
from matplotlib import pyplot as plt
from time import strftime

plt.style.use('ggplot')
# from pandas.plotting import autocorrelation_plot
# from statsmodels.graphics import tsaplots
import platform


# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 8,               # LaTeX default is 10pt font.
#     "font.size": 8,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 5,
#     "ytick.labelsize": 5,
#     "figure.figsize": [4,4],     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)
class Plotting():

    def __init__(self, data, pred, idx, address=None, lag=None, savedata=True, saveplots=True):
        """

        :param data: :type torch.tensor() The tensor should contain the data that is going to be plotted.
                                          the last column represents the objective training.
        :param idx: :type list of ints :description specifies the indices that need to plotted.
        :param address :type string  :descrp The string of the address to be plotted
        :param lag:
        :param savedata: :type bool :descrp True save data, else False
        :param saveplots: :type bool :descrp True save plots, else False

        """
        self.samples = data
        self.predSamples = pred
        self.lag = lag
        self.PATH = '../plots/'
        self.addresss = address
        self.idx  = idx
        os.makedirs(self.PATH, exist_ok=True)
        self.PATH_fig = os.path.join(self.PATH)
        os.makedirs(self.PATH_fig, exist_ok=True)
        self.PATH_data = '../data'
        os.makedirs(self.PATH_data, exist_ok=True)


    def plot_hist(self, kdeplot=False, klLoss=False, independent=True):
        """
        Plots histogram and density estimate if required.
        Typically we will have the data for one batch, plus the prediction.
        But users may want the density plots on separate or different scripts.
        TODO: Create script that concats batches of data and then combines them for plotting.

        :param kdeplot :type: bool Whether to include the density plot.
        :param klLoss :type bool  If the last column contains the objective information, then True, else False.
        :param independent :type bool True if you want each plot on a different subplot. Else, plots on one figure.
        :return:
        """
        plt.clf()

        # plot prediction, original output data, comparison plot
        if independent:
            numSubplots = 3
        else:
            numSubplots = 1
        dataPred = self.predSamples.numpy()
        dataGen = self.samples[self.idx].numpy()





        if klLoss:
            numSubplots += 1
        g = sns.FacetGrid(self.samples, col="y")
        g.map(sns.distplot, self.samples, bins='auto', norm_hist=True, kde=kdeplot)
        g.add_legend()
        fname = '.pdf'
        path_image = self.PATH_fig + '/' + fname
        # weights = np.ones_like(self.samples) / len(self.samples)
        fig_width = 3.39  # width in inches
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches
        plt.figure(figsize=(fig_width, fig_height))
        sns.distplot(self.samples, bins='auto', norm_hist=True, kde=kde)

        plt.savefig(os.path.join(self.PATH_fig, fname))

    def plot_density(self, all_on_one=True):
        """
        Plots either all the histograms for each param on one plot, or does it indiviually
        dependent on keys
        :param all_on_one type: bool
        :param keys type:list
        :return:
        """

        if all_on_one:
            fname = 'density_plot_of_parameters.pdf'
            path_image = self.PATH_fig + '/' + fname
            self.samples.plot(subplots=True, kind='kde')
            plt.savefig(os.path.join(self.PATH_fig, fname))
            print(50 * '=')
            print('Saving histogram w/o burnin plot to {0}'.format(path_image))
            print(50 * '=')

    def auto_corr(self):
        """
        Plots for each parameter the autocorrelation of the samples for a specified lag.
        :return:
        """
        x = {}
        keys = copy.copy(self.keys)
        fig, axes = plt.subplots(ncols=len(self.keys), sharex=True, sharey=True)
        fig.text(0.5, 0.04, 'lag', ha='center', va='center')
        fig.text(0.02, 0.5, 'autocorrelation', ha='center', va='center', rotation='vertical')
        # fig.suptitle('Autocorrelation')
        for key in self.keys:
            x[key] = []
            for i in range(self.lag):
                x[key].append(self.samples[key].autocorr(lag=i))
        if len(self.keys) == 1:
            key = keys.pop()
            axes.stem(np.arange(0, len(x[key]), step=1), x[key], linestyle='None', markerfmt='.', markersize=0.2,
                      basefmt="None", label=key)
            axes.legend(loc='upper right')
        else:
            for axis in axes.ravel():
                # https: // stackoverflow.com / questions / 4700614 / how - to - put - the - legend - out - of - the - plot
                key = keys.pop()
                axis.stem(np.arange(0, len(x[key]), step=1), x[key], linestyle='None', markerfmt='.', markersize=0.2,
                          basefmt="None", label=key)
                axis.legend(loc='upper right')

        fname2 = 'Autocorrelationplot.pdf'
        plt.savefig(os.path.join(self.PATH_fig, fname2), dpi=400)
        path_image = self.PATH_fig + '/' + fname2
        print(50 * '=')
        print('Saving  autocorrelation plots to: {0}'.format(path_image))
        print(50 * '=')
