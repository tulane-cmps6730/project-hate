from flask import render_template, flash, redirect, session
from sentence_transformers import SentenceTransformer
from . import app
from .forms import MyForm
from .. import clf_path
from ..cli import clf
import torch
import pickle
import sys
import torch

clf.load_state_dict(torch.load(clf_path))
print('read clf %s' % str(clf))
#print('read vec %s' % str(vec))
labels = ['implicit_hate','not_hate','explicit_hate']
sbert384 = SentenceTransformer("all-MiniLM-L6-v2")

##@app.route('/index', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
		X = torch.tensor(sbert384.encode(input_field),dtype=torch.float64)
		proba = clf.forward(X)
		print("Probability Not Hate: "+str(proba[0].item())+"\nProbability Implicit Hate: "+str(proba[1].item())+"\nProbability Explicit Hate: "+str(proba[2].item()))
		pred = torch.argmax(proba)
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=labels[pred], confidence='%.2f' % proba[pred])
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
