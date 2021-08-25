import sys
from os.path import dirname, abspath, join
import sys
from config import config
# Find code directory relative to our directory"
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', 'src'))
sys.path.append(CODE_DIR)
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from transformers import TFBertForSequenceClassification, BertTokenizer
import warnings

warnings.filterwarnings('ignore', message='foo bar')


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = "regression_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline


def load_Model(label2id=None, id2label=None):
    """:cvar"""
    if label2id is None and id2label is None:
        tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_DIR)
        model = TFBertForSequenceClassification.from_pretrained(config.MODEL_DIR)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_DIR)
        model = TFBertForSequenceClassification.from_pretrained(config.MODEL_DIR,
                                                                id2label=id2label,
                                                                label2id=label2id,
                                                                num_labels=len(label2id),
                                                                ignore_mismatched_sizes=True)

    return tokenizer, model


def load_train_data():
    df = pd.read_csv(config.DATA_PATH)
    id2lable = joblib.load(config.ID2LABEL_PATH)
    label2id = joblib.load(config.LABEL2ID_PATH)
    before = len(id2lable)
    f = len(label2id)
    for i in df['cname']:
        if i not in label2id:
            label2id[i] = f
            id2lable[f] = i
            f = f + 1
    after = len(id2lable)
    if(before!=after):
        joblib.dump(label2id,config.LABEL2ID_PATH)
        joblib.dump(id2lable,config.ID2LABEL_PATH)
    else:
        print("Same classes")
    df["labels"] = df['cname'].map(label2id)
    return df, label2id, id2lable


def load_prediction_data():

    return pd.read_csv(config.DATA_PATH)

