# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys
import numpy as np
import pandas as pd
import re
import requests
import torch
from sentence_transformers import SentenceTransformer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from . import clf_path, config
from . import model as clf#  import model


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
    df = data2df()
    print(df.columns)
    train_X1 = torch.tensor(df[df.columns[2:]].to_numpy(),dtype=torch.float64)
    #train_X1 = z(train_X1)
    train_Y1 = []
    for c in df['class']:
        y_hot = [0,0,0]
        y_hot[int(['implicit_hate', 'not_hate', 'explicit_hate'].index(c))] = 1
        y_hot = torch.tensor(y_hot,dtype=torch.float64)
        train_Y1.append(y_hot)
    m1train_data1 = list(zip(train_X1,train_Y1))
    
    losses = clf._train(m1train_data1,1000,0.01)
    print(losses)
    torch.save(clf.state_dict(),clf_path)
    return

def z(x):
    n = x.shape[0]
    mean = torch.sum(x,dim=0)/n # X.mean()
    std = torch.std(x,dim=0)
    z_scores = (x - mean)/std
    return z_scores

if __name__ == "__main__":
    sys.exit(main())
