# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:05:07 2020

@author: Ksenia Mukhina
"""
import datetime
import os
import torch

#Sort file names according to dates in titles
def natural_keys(text):
    date = [int(x) for x in text.split('.')[0].split('-')[1:]]
    value = int(datetime.datetime(*date).timestamp())
    return value

#Loads state dictionary for model and optimizer, loss' history, and current epoch
def load_checkpoint(model, optimizer, filename='checkpoint-gpu.pth.tar'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("loading checkpoint '{}'".format(filename))
        
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses = checkpoint['losses']
        error_fields_list = checkpoint['error_fields_list']
        
        print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losses, error_fields_list

#Weighted average of 2-dimensional tensors
def create_weighted_average_field(fields_list):
    tmp = []
    n = len(fields_list) 
    for i in range(n):
        t = torch.div(fields_list[i] * (i + 1), n * (n+1) / 2 )
        tmp.append(t)
        
    error_field = torch.stack(tmp)
    return error_field.sum(0)

#Initilize model weights and biases
def init_normal(m, ksi = True, alex = True):
    if ksi:
        m.ksi_fc2.bias.data.fill_(0.5)
        
    if alex:
        m.alex_fc.bias.data.fill_(0.5)
    
    if type(m) == torch.nn.Conv3d:
        torch.nn.init.dirac_(m.weight)
