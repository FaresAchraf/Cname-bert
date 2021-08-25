import sys
from os.path import dirname, abspath, join
import sys
import warnings

warnings.filterwarnings('ignore', message='foo bar')

# Find code directory relative to our directory"
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', 'src'))
sys.path.append(CODE_DIR)

import BertCname.pipeline as pipe

import BertCname.processing.data_management as dm


def run_training():
    df, label2id, id2lable = dm.load_train_data()
    tokenizer, model = dm.load_Model(id2label=id2lable, label2id=label2id)
    classifier_pipe = pipe.create_pipeline(model, tokenizer)
    classifier_pipe.fit(df, df["labels"])


if __name__ == '__main__':
    run_training()
