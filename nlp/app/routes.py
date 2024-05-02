from flask import render_template, flash, redirect, session
from sentence_transformers import SentenceTransformer
from . import app
from .forms import MyForm
from .. import clf_path
import torch
import pickle
import sys

clf = pickle.load(open(clf_path, 'rb'))
print('read clf %s' % str(clf))
#print('read vec %s' % str(vec))
labels = ['not_hate', 'implicit_hate','explicit_hate']
sbert384 = SentenceTransformer("all-MiniLM-L6-v2")

##@app.route('/index', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
		X = sbert384.encode(input_field)
		proba = clf.forward(X)
		print("Probability Not Hate: "+proba[0]+"\nProbability Implicit Hate: "+proba[1]+"\nProbability Explicit Hate: "+proba[2])
		proba = torch.argmin(proba)
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction="Most Likely: "+labels[pred], confidence='%.2f' % proba)
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
