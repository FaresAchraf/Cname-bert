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

DATA_PATH = abspath(join(THIS_DIR, 'data', 'categorical_data.csv'))


def get_prediction():
    """
    this is a prediction function
    :param text: input text
    :return: tuple
    """
    df = dm.load_prediction_data()
    tokenizer,model = dm.load_Model()
    classifier_pipe = pipe.create_pipeline(model, tokenizer)
    return classifier_pipe.predict(df)


if __name__ == '__main__':
    # Create the parser
    #parser = argparse.ArgumentParser()
    # Add an argument
    #parser.add_argument('--text', type=str, required=True)
    # Parse the argument
    #args = parser.parse_args()
    #print(get_prediction([args.text]))

    print(get_prediction().head())
