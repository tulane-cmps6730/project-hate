# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """A Student"""
__email__ = 'mmontgomery1@tulane.edu'
__version__ = '0.1.0'##I guess?

# -*- coding: utf-8 -*-
import configparser
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
	w = open(path, 'wt')
	w.write('[data]\n')#<a data-resin-target="openfile" class="item-link item-link " href="/file/1519554536417">implicit_hate_v1_stg1_posts_embedded.tsv</a>
	w.write('url = https://tulane.box.com/s/1ru8jihn2vi8qpruzbgc7dq73pnnqj4u')#https://app.box.com/index.php?rm=box_download_shared_file&shared_name=1ru8jihn2vi8qpruzbgc7dq73pnnqj4u&file_id=f_1519554536417\n')
	w.write('file = %s%s\n' % (nlp_path, 'hate_speech_dataset_embeddings.tsv'))
	w.close()

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['PWD'] + os.path.sep + 'nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
clf_path = nlp_path + 'clf.pth'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)
print(nlp_path)

class IDHate_simple(nn.Module):
    def __init__(self):
        super(IDHate_simple,self).__init__()
        self.softmax = nn.Softmax(dim=0)#Need this for the probabilities at the end
        #self.sigmoid = nn.Sigmoid()#Need this to normalize at each layer
        #self.tanh = nn.Tanh()#Need this for weights; want to be able to weight a component but in a way that doesn't affect the classification
        self.W = nn.Parameter(torch.zeros((384,3),dtype=torch.float64),requires_grad=True)#used to make X-based weights
        self.S = torch.zeros((1,3),requires_grad=False,dtype=torch.float64)
        self.labels = ['implicit_hate', 'not_hate', 'explicit_hate']
        self.output = torch.zeros((1,3))
        return
    
    def forward(self,x):        
        self.S= x @ self.W
        self.output = self.softmax(self.S)
        return self.output
    
    def pretrained(self):
        self.load_state_dict(torch.load('/Users/temp/Documents/Spring_2024/NLP/mmontgomery1-master/project/model1_neural.pth'))
        return
    def _print(self):
        print("W","\n",self.W)
        print("S","\n",self.S)
        print("p","\n",self.output)
        return
    def _train(self,data,epochs,learning_rate):
        criterion = nn.CrossEntropyLoss()
        torch.random.manual_seed(42)  
        np.random.seed(42)
        optimizer = torch.optim.Adam(self.parameters(),
                                        lr=learning_rate)
        sched = StepLR(optimizer,gamma=0.3,step_size=10)
        loss_val = []
        # main training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            np.random.shuffle(data)
            losses = []
            for batch, (X, y) in enumerate(data[:300]):
                #print(y)
                result = self.forward(X)
                loss = criterion(result,y)
                loss.backward()      # computes all the gradients
                optimizer.step()
                losses.append(loss.item())
            loss_val.append(np.mean(losses))
            sched.step()
        return loss_val

model = IDHate_simple()