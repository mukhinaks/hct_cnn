# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:54:32 2020

@author: Ksenia Mukhina
"""
from torch.utils.data import DataLoader
import torch
from torch.utils import data
import os
import pickle
import misc.helper

class Dataset(data.Dataset):
    def __init__(self, source_path, windows, level):
        #Loads all paths
        for root, dirs, files in os.walk(source_path):
            files.sort(key=misc.helper.natural_keys)
            all_files = [ os.path.join(root, file) for file in files] 
        
        self.grids = []
        self.window = windows
        self.level = level
        self.split = 10
        self.prediction_hour = 1
        self.shift = 24 * 7
        
        for f in range( self.shift, len(all_files)):
            
            idxs = [x for x in range(f - self.window - self.prediction_hour + 1, f + 1)]
            idxs.append(f - self.shift)
            
            if level == 1:
                tmp_data = []
                for g in idxs:                
                    with open(all_files[g], 'rb') as file:
                        tmp = pickle.load(file)

                    data = {k: v['sum'] for k, v in tmp.items()}
                    tmp_data.append( data)

                hour_data = tmp_data[:self.window]
                week_data = tmp_data[-1]
                label = tmp_data[-2]

                self.grids.append( (hour_data, week_data, label, {1: all_files[f]} ) )
                
            elif level == 2:
                with open(all_files[f], 'rb') as file:
                        tmp_hour = pickle.load(file)
                
                tmps = []
                for g in idxs:                
                    with open(all_files[g], 'rb') as file:
                            tmp = pickle.load(file)
                            
                    tmps.append(tmp)
                
                for i in range(self.split):
                    for j in range(self.split):
                        if (i,j) in tmp_hour:
                            if tmp_hour[(i,j)]['sum'] > 10:
                                tmp_data = []
                                for tmp in tmps:                                                    
                                    if (i,j) in tmp:
                                        data = {k_1: v_1['sum'] for k_1, v_1 in tmp[(i,j)]['data'].items()}
                                    else:
                                        data = {}
                                    tmp_data.append( data)
                                    
                                hour_data = tmp_data[:self.window]
                                week_data = tmp_data[-1]
                                label = tmp_data[-2]

                                self.grids.append( (hour_data, week_data, label, {1: all_files[f], 2: (i,j)}) )   
            elif level == 3:
                with open(all_files[f], 'rb') as file:
                        tmp_hour = pickle.load(file)
                tmps = []        
                for g in idxs:                
                    with open(all_files[g], 'rb') as file:
                         tmp = pickle.load(file)
                    tmps.append(tmp)
                        
                for i in range(self.split):
                    for j in range(self.split):
                        if (i,j) in tmp_hour:
                            if tmp_hour[(i,j)]['sum'] > 10:
                                
                                for i_1 in range(self.split):
                                    for j_1 in range(self.split):
                                        if (i_1,j_1) in tmp_hour[(i,j)]['data']:
                                            if tmp_hour[(i,j)]['data'][(i_1,j_1)]['sum'] > 10:
                                                
                                                tmp_data = []
                                                for tmp in tmps:                                                                    
                                                    if (i,j) in tmp:
                                                        if (i_1,j_1) in tmp[(i,j)]['data']:
                                                            data = {
                                                                k_1: v_1 for k_1, v_1 in tmp[(i,j)]['data'][(i_1,j_1)]['data'].items()
                                                            }
                                                        else:
                                                            data = {}
                                                    else:
                                                            data = {}
                                                    tmp_data.append( data)

                                                hour_data = tmp_data[:self.window]
                                                week_data = tmp_data[-1]
                                                label = tmp_data[-2]

                                                self.grids.append( (hour_data, week_data, label, {
                                                    1: all_files[f], 
                                                    2: (i,j),
                                                    3: (i_1,j_1)
                                                }) )    
       

    def __len__(self):
        return len(self.grids) 

    def get_data(self, data):            
        i = torch.LongTensor(list(data.keys()))
        v = torch.FloatTensor(list(data.values()))
            
        try:
                tmp = torch.sparse.FloatTensor(i.t(), v, torch.Size([self.split,self.split])).to_dense() 
        except:
                tmp = torch.zeros([self.split, self.split])
                
        return tmp
                
    def __getitem__(self, index):   
        #Creates sample
        hour_data, week_data, label, cell_address = self.grids[index]
      
        input_tensors = []

        for x in hour_data:
            input_tensors.append(self.get_data(x) )
          
        #Forms data and its label
        X = torch.stack(input_tensors)
        X_24 = self.get_data(week_data).unsqueeze(0) 
        
        y = self.get_data(label) 

        return (X, X_24), y, cell_address