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
    """

    Function to load and train the pipeline and save the model weights and config in new_model directory
    """
    df, label2id, id2lable = dm.load_train_data()
    tokenizer, model = dm.load_model(id2label=id2lable, label2id=label2id)
    classifier_pipe = pipe.create_pipeline(model, tokenizer)
    trained_pipe = classifier_pipe.fit(df, df["labels"])
    trained_pipe['Model'].model.save_pretrained('new_model')


if __name__ == '__main__':
    run_training()
