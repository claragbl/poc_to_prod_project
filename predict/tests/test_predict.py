import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run as run_train
from keras.models import load_model

import os
import json
import time

from preprocessing.preprocessing import utils
from predict.predict import run


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestPredict(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        # create a dictionary params for train conf
        params = {'batch_size': 2,
                  'epochs': 1,
                  'dense_dim': 64,
                  'min_samples_per_label': 1,
                  'verbose': 1}

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # TRAIN
            _, artefact_path = run_train.train(dataset_path="hfhzkgzhf.csv", train_conf=params, model_path=model_dir, add_timestamp=True)

            # load model
            model = load_model(os.path.join(artefact_path, 'model.h5'))

            # load params
            with open(os.path.join(artefact_path, 'params.json'), 'r') as params:
                params = json.load(params)

            # load labels_to_index
            with open(os.path.join(artefact_path, 'labels.json'), 'r') as labels:
                idx_to_label = json.load(labels)
        print(idx_to_label)
        # PREDICT
        model = run.TextPredictionModel(model, params, idx_to_label)
        predictions = model.predict(['Is it possible to execute the procedure of a function in the scope of the caller?'], 1)
        print(predictions[0][0])
        result = idx_to_label[str(predictions[0][0])]
        print(result)

        # ASSERT
        self.assertEqual(result, "php")
