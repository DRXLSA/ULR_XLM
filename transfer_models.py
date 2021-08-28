




# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:51:18 2021

@author: lenovo
"""
import torch
import torch.nn as nn

class transfer_model(nn.Module):
    
    def __init__(self,layer_num,inp_dim=1024,emb_dim=1024):
        super(transfer_model,self).__init__()
        self.layer_num = layer_num
        self.emb_dim = emb_dim
        self.layers = nn.ModuleList([])
        inp_dim = 1024
        for i in range(layer_num):

            if i == layer_num-1:
                self.layers.add_module('linear{}'.format(i),module=nn.Linear(inp_dim,self.emb_dim))
            elif i==0:
                self.layers.add_module('linear{}'.format(i),module=nn.Linear(inp_dim,self.emb_dim*2))
                self.layers.add_module('activate{}'.format(i),module=nn.LeakyReLU(0.2))
                self.layers.add_module('dropout{}'.format(i),module=nn.Dropout(0.1))
                inp_dim = self.emb_dim*2
            else:
                self.layers.add_module('linear{}'.format(i),module=nn.Linear(self.emb_dim*2,self.emb_dim*2))
                self.layers.add_module('activate{}'.format(i),module=nn.LeakyReLU(0.2))
                self.layers.add_module('dropout{}'.format(i),module=nn.Dropout(0.1))
                inp_dim = self.emb_dim*2
    
    def forward(self,inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp
    
    
if __name__ == '__main__':
    inp =torch.ones(768)
    model = transfer_model(3)
    res = model(inp)
