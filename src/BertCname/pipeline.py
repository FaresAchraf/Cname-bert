from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', 'src'))
print(CODE_DIR)
sys.path.append(CODE_DIR)

from sklearn.pipeline import Pipeline

from BertCname.processing import  preprocessors as pp
from transformers import pipeline


def create_pipeline(model, tokenizer=None):
    classifier_pipe = Pipeline([
                            ("RemovePunct",pp.RemovePunct()),
                            ("RM_ASCII",pp.RM_ASCII()),
                            ("RmDigits_Lower",pp.RmDigits_Lower()),
                            ("Tokenizer",pp.Tokenizer(tokenizer=tokenizer)),
                            ("Model",pp.Model(model=model))
                            #("classifier",pp.Classifier(pipe=pipeline(model=model,tokenizer=tokenizer,task='sentiment-analysis')))
    ])
    return classifier_pipe
