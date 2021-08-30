from os.path import dirname, abspath, join
import sys
import warnings

warnings.filterwarnings('ignore', message='foo bar')

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../', 'src'))
sys.path.append(CODE_DIR)

DATA_PATH = abspath(join(THIS_DIR, '../','data', 'categorical_data.csv'))
WEIGHTS_DIR = abspath(join(THIS_DIR, '../', 'trained_model','model','tf_model.h5'))
TOKENIZER_DIR = abspath(join(THIS_DIR, '../', 'trained_model','tokenizer'))
CONF_DIR = abspath(join(THIS_DIR, '../', 'trained_model','model','config.json'))
MODEL_DIR = abspath(join(THIS_DIR, '../', 'trained_model','model'))
ID2LABEL_PATH = abspath(join(THIS_DIR, 'id2label.json'))
CLASS2CATEGORYID_PATH = abspath(join(THIS_DIR, 'class2category_id.json'))
LABEL2ID_PATH = abspath(join(THIS_DIR, 'label2id.json'))
LABEL2ID = dict()