#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:30
Date created:  02/09/2019

License: MIT
'''

import torch
import torch.distributions as dist
import warnings

class Proposal():

    def __init__(self, *args, **kwargs):
        args = args
        kwargs = kwargs

    def Normalapproximator(self, *args, **kwargs):
        '''

        :param inData: :type torch.tensor The data that you want to regress over to learn amortized function
        :param batchSize: :type int batch size
        :param model: neural network
        :return: prediction to the regression problem specified by inData
        '''

        return dist.Normal(*kwargs['model'](kwargs['inData']))
