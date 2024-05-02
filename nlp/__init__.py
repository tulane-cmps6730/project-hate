# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """A Student"""
__email__ = 'mmontgomery1@tulane.edu'
__version__ = '0.1.0'##I guess?

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
	w = open(path, 'wt')
	w.write('[data]\n')#<a data-resin-target="openfile" class="item-link item-link " href="/file/1519554536417">implicit_hate_v1_stg1_posts_embedded.tsv</a>
	w.write('url = https://app.box.com/index.php?rm=box_download_shared_file&shared_name=1ru8jihn2vi8qpruzbgc7dq73pnnqj4u&file_id=f_1519554536417\n')
	w.write('file = %s%s%s\n' % (nlp_path, os.path.sep, 'hate_speech_dataset_embeddings.tsv'))
	w.close()

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['HOME'] + os.path.sep + '.nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
clf_path = nlp_path + 'clf.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)
print(config)