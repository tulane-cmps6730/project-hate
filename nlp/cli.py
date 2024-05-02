# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys
import torch
import numpy as np
import pandas as pd
import re
import requests
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import StepLR
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from . import clf_path, config


@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
@main.command('dl-data')
def dl_data():
    """
    Download training/testing data.
    """
    data_url = config.get('data', 'url')
    data_file = config.get('data', 'file')
    print('downloading from %s to %s' % (data_url, data_file))
    r = requests.get(data_url)
    with open(data_file, 'wt') as f:
        f.write(r.text)
    

def data2df():
    return pd.read_csv(config.get('data', 'file'),sep="\t")

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()
    print('%d rows' % len(df))
    print('label counts:')
    print(df['class'].value_counts())    

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """    
    class IDHate_simple(nn.Module):
        def __init__(self):
            super(IDHate_simple,self).__init__()
            self.softmax = nn.Softmax(dim=0)#Need this for the probabilities at the end
            self.sigmoid = nn.Sigmoid()#Need this to normalize at each layer
            self.tanh = nn.Tanh()#Need this for weights; want to be able to weight a component but in a way that doesn't affect the classification
            self.W = nn.Parameter(torch.zeros((384,3),dtype=torch.float64),requires_grad=True)#used to make X-based weights
            self.S = torch.zeros((1,3),requires_grad=False,dtype=torch.float64)
            self.labels = ['implicit_hate', 'not_hate', 'explicit_hate']
            self.output = torch.zeros((1,3))
            return
        
        def forward(self,x):        
            self.S= x @ self.W
            self.output = self.softmax(self.S)
            return self.output

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
                    loss = criterion(result[0],y)
                    loss.backward()      # computes all the gradients
                    optimizer.step()
                    losses.append(loss.item())
                loss_val.append(np.mean(losses))
                sched.step()
            return
    df = data2df()
    clf = IDHate_simple()
    clf._train(df)
    pickle.dump((clf), open(clf_path, 'wb'))
    return

"""def do_cross_validation(clf, X, y):
    all_preds = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X,y):
        clf.fit(X[train], y[train])
        all_preds[test] = clf.predict(X[test])
    print(classification_report(y, all_preds))    """

"""def top_coef(clf, vec, labels=['not_hate', 'implicit_hate', "explicit_hate"], n=10):
    feats = np.array(vec.get_feature_names_out())
    print('top coef for %s' % labels[1])
    for i in np.argsort(clf.coef_[0])[::-1][:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
    print('\n\ntop coef for %s' % labels[0])
    for i in np.argsort(clf.coef_[0])[:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
"""
if __name__ == "__main__":
    sys.exit(main())
