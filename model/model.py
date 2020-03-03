# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:16:05 2020

@author: Ksenia Mukhina
"""
import torch
class PredictionModel(torch.nn.Module):
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.ksi_cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(1, 32, kernel_size=3, stride=1,),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, ),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(64, 128, kernel_size=(1,3,3), stride=1,),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(256, 128, kernel_size=(1,3,3), stride=(1,1,1), ),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(128, 64, kernel_size=(1,3,3), stride=(1,1,1),),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((1,1,1,1,0,0)),
            torch.nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(1,1,1), ),
            torch.nn.ELU(),
            torch.nn.Conv3d(32, 1, kernel_size=1),
            torch.nn.ELU()
        )

        self.ksi_fc1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=1),
            torch.nn.ELU(),
        )
        self.ksi_fc2 = torch.nn.Conv2d(1, 1, kernel_size=1)
        
        self.alex_branch_hours = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3, stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(64, 128, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=1,),
            torch.nn.ELU(),
        )
        
        self.alex_branch_day = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(32, 64, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(64, 128, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
        )
        
        self.alex_branch_attention = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(32, 64, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(64, 128, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
        )
        
        self.alex_cnn_merge = torch.nn.Sequential(
            torch.nn.ReplicationPad3d((2,2,2,2, 0, 0)),
            torch.nn.Conv3d( 512 + 256, 256, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((2,2,2,2, 0, 0)),
            torch.nn.Conv3d(256, 128, kernel_size=(1,3,3), stride=1,),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((2,2,2,2, 0, 0)),
            torch.nn.Conv3d(128, 64, kernel_size=(1,3,3), stride=1, ),
            torch.nn.ELU(),
            torch.nn.ReplicationPad3d((2,2,2,2, 0, 0)),
            torch.nn.Conv3d(64, 8, kernel_size=(1,3,3), stride=1,),
            torch.nn.ELU(),
            torch.nn.Conv3d(8, 1, kernel_size=1),
            torch.nn.ELU(),
        )
        
        self.alex_error_attention = torch.nn.Sequential(
            torch.nn.Conv3d(1, 1, kernel_size=1),
            torch.nn.ELU(),
        )
        
        self.alex_fc = torch.nn.Conv2d(2, 1, kernel_size=1)
        
        self.final_fc = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=1),
        )
        
        
    def forward(self, time_windows, previous_day, error_field):
        ksi_x1 = self.ksi_cnn(time_windows.unsqueeze(1))
        ksi_x2 = previous_day.unsqueeze(1)
        
        ksi_x = torch.cat((ksi_x1, ksi_x2), dim=1)
        ksi_x = self.ksi_fc1(ksi_x.squeeze(2))
        ksi_x = self.ksi_fc2(ksi_x)
        
        alex_x1 = self.alex_branch_hours(time_windows.unsqueeze(1))
        alex_x2 = self.alex_branch_day(previous_day.unsqueeze(1))
        alex_x3 = self.alex_branch_attention(error_field.unsqueeze(1))
        
        alex_x4 = self.alex_error_attention(time_windows[:, -1, :, :].unsqueeze(1).unsqueeze(1)).squeeze(1)
        
        alex_x = torch.cat((alex_x1, alex_x2, alex_x3), dim=1)
        alex_x = self.alex_cnn_merge(alex_x).squeeze(1) 
        
        alex_x = torch.cat((alex_x, alex_x4), dim=1)
        alex_x = self.alex_fc(alex_x) 
        
        x = torch.cat((alex_x, ksi_x), dim=1)
        x = self.final_fc(x)
        
        return x
