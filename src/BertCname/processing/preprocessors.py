import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import time
from datetime import date
import joblib
from config import config
# Remove punctuation Class
class RemovePunct(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_punct(self, text):
        """

        :param text:
        :return: Text without punctuation
        """
        nopunct = ""
        for c in text:
            if c not in string.punctuation:
                nopunct = nopunct + c
        return nopunct

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """

        :param x:
        :return: apply the remove_puunct to  all the product names
        """
        x['pname'] = x['pname'].apply(self.remove_punct)
        return x


# Remove Non ASCII text Class
class RmAscii(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """

        :param x: input DataFrame
        :return: DataFrame with removed non ASCII characters
        """
        x['pname'] = x['pname'].apply(lambda y: re.sub(r'[^\x00-\x7F]+', "", y))
        return x


## RemoveDigit Class
class RmDigitsLower(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        x['pname'] = x['pname'].apply(lambda y: re.sub('\w*\d\w*', '', y.lower()))
        return x


# Classifier Class
class Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, *, pipe):
        self.pipe = pipe
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self

    def predict(self, X):
        return self.pipe(X)


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, *, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> (pd.DataFrame,dict):
        """

        :param X: pd.DataFrame ( preprocessed data )
        :return: pd.DataFrame ( preprocessed data ) , dict (Tokenized data)
        """

        return X, self.tokenizer(list(X["pname"]), padding=True, truncation=True, return_tensors="tf").data


class Model(BaseEstimator, TransformerMixin):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X: tuple) -> pd.DataFrame:
        """"

        :param X: tuple ( input dataframe, tokenized data )
        :return: pd.DataFrame ( preprocessed data ) , dict (Tokenized data)
        """
        prediction = self.model.predict(X[1]).logits
        prediction = tf.math.softmax(prediction)
        idx = np.argmax(prediction, axis=-1)
        df = pd.concat([pd.DataFrame({'category_id': list(map(lambda x: (self.model.config.id2label.get(x)), list(idx))),
                                        "score": list(np.max(prediction, axis=-1))}), X[0]], axis=1)
        df['algo_name']="Bert_Cname"
        t = time.localtime()
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        current_time = time.strftime("%H:%M", t)
        df['created'] = str(d1)+" "+current_time
        df['algo_version'] = "1.0.0"
        return df[['category_id', "score", "created", "algo_name", "algo_version"]]


    def fit(self, X, y=None):

        """

        :param X: tuple ( input dataframe, tokenized data )
        :param y: labels
        :return: TFBertModel
        """
        epochs = 4
        batch_size = 32
        num_training_steps = (len(X[1]['attention_mask']) // batch_size) * epochs
        lr_scheduler = PolynomialDecay(
            initial_learning_rate=5e-5,
            end_learning_rate=0,
            decay_steps=num_training_steps
        )
        opt = Adam(learning_rate=lr_scheduler)
        self.model.compile(
            optimizer=opt,
            loss=self.model.compute_loss,
            metrics=['accuracy']
        )
        self.model.fit(X[1], y)

        return self.model
