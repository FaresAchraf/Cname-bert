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


def create_pipeline(model, tokenizer=None) -> Pipeline:
    """

    :param model: pretrained Bert model
    :param tokenizer: pretrained tokenizer
    :return: Pipeline for prediction or training
    """
    classifier_pipe = Pipeline([
                            ("RemovePunct", pp.RemovePunct()),
                            ("RM_ASCII", pp.RmAscii()),
                            ("RmDigits_Lower", pp.RmDigitsLower()),
                            ("Tokenizer", pp.Tokenizer(tokenizer=tokenizer)),
                            ("Model", pp.Model(model=model))
    ])
    return classifier_pipe
