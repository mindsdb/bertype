""" embedders.py
    Implements several column embedding schemes.
"""
import os
import re
import datetime

import pandas
import numpy

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import SentencesDataset
from sentence_transformers.models import Transformer
from sentence_transformers.models import Pooling
from sentence_transformers.models import CNN
from sentence_transformers.models import Dense
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from bertype.column_info import get_types
from bertype.column_info import get_subtypes


def multiple_replace(replacements, text):
    """ Replaces multiple instances in `text`.

        :param replacements (dict)
            dictionary with keywords to replace as keys,
            and values to replace as values
        :param text (string)
            string to operate on.

        :returns
            original string with replacements made.
    """
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacements[mo.group()], text)


# utilitary function
def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss

        :param elapsed (int)
            time in seconds since arbitrary event.
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    r = str(datetime.timedelta(seconds=elapsed_rounded))

    return r


# utilitary function
def flat_accuracy(preds, labels):
    """ Calculate accuracy between predictions and labels

        :param preds (numpy.ndarray, float32)
            2D array of shape (B, N) where B is the batch size and
            N is the number of categories. `preds` must be the output
            of a `softmax` layer.
        :param labels (numpy.ndarray, int)
            1D array of actual labels as integers.

        :returns (float)
            accuracy calculated as number of hits divided
            by the total number of provided labels
    """
    pred_flat = numpy.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    r = numpy.sum(pred_flat == labels_flat) / len(labels_flat)

    return r


class ColBert:
    """ Implements column embeddings using BERT.

        Creates embeddings that are different for columns of
        different types; do not confuse with subtypes or modifiers,
        which are handled differently.
    """
    __DEFAULT_WORD_EMBEDDING_MODEL__ = 'sentence-transformers/all-distilroberta-v1'

    def __init__(self,
                 device: str = 'cpu',
                 model_name: str = None):
        """ Initializer
        """
        self.emb_size = -1
        self.final_emb_size = 32
        self.max_length = 512
        self.batch_size = 24

        # type encoder
        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(
            numpy.asarray(get_types()))
        self.num_data_types = len(get_types())
        # sub-type encoder
        self.subtype_encoder = LabelEncoder()
        self.subtype_encoder.fit(
            numpy.asarray(get_subtypes()))
        self.num_data_types = len(get_subtypes())
        self.device = torch.device(device)

        if model_name is None:
            # use scibert by default
            self.word_embedding_model = Transformer(
                self.__DEFAULT_WORD_EMBEDDING_MODEL__,
                max_seq_length=self.max_length
            )
            self.emb_size = \
                self.word_embedding_model.get_word_embedding_dimension()
            # mean pooling instead of average pooling with normalization
            self.pooling_model = Pooling(
                self.emb_size,
                pooling_mode='max')
            # convolutional head to better capture non-linear
            # relationships and allow effective reduction of
            # embedding dimension from 768 to just 16
            self.cnn = CNN(self.emb_size, out_channels=256, kernel_sizes=[3, 5, 7, 11])
            # two linear layers to capture some non-linearity
            # in dimensionality reduction. Use LeakyReLY activation
            self.fc1 = Dense(1024, 128, activation_function=torch.nn.Identity())
            self.fc2 = Dense(128, self.final_emb_size, activation_function=torch.nn.Identity())
            # normalization layer
            # self.norm_layer = Normalize()
            self.model = SentenceTransformer(
                modules=[
                    self.word_embedding_model,
                    self.cnn,
                    self.pooling_model,
                    self.fc1,
                    self.fc2,
                    # self.norm_layer,
                ]
            )
        else:
            self.load(model_name)
        self.sentence_list = []
        self.label_list = []

        self.train_dataloader = DataLoader
        self.train_dataloader_2 = DataLoader
        self.evaluator = EmbeddingSimilarityEvaluator

    def get_embedding_length(self) -> int:
        """ Returns length of embedding.
        """
        return self.final_emb_size

    def decode_subtype(self, col_type_codes):
        """ Transforms column sub-type codes to names.
        """
        x = numpy.atleast_1d(col_type_codes)
        x = x.astype(numpy.int32)
        col_type_names = self.subtype_encoder.inverse_transform(x)
        return col_type_names.ravel()[0]

    def decode_type(self, col_type_codes):
        """ Transforms column type codes to names.
        """
        x = numpy.atleast_1d(col_type_codes)
        x = x.astype(numpy.int32)
        col_type_names = self.type_encoder.inverse_transform(x)
        return col_type_names.ravel()[0]

    def column_to_sentences(self, column_data: pandas.Series):
        """ Converts a series into chunks of smaller sentences.

        :param column_data (pandas.Series)
            original data from DataFrame.
        :return sentences (list)
            contains the data of `column_data` as chunks of strings.

        :note
            This method might be a little too slow for very
            large columns, but it is actually intended to work
            for 10k to 100k - ish entries, for which it is
            fast enough.
        """
        ready = False
        sentence = ''
        sentences = []
        # replace *actual* NaN/NaT/None with missing keyword
        column_data = column_data.fillna('__missing__')
        # trucate entires to have 96 characters.
        column_data = column_data.astype(str).str.slice(0, 96)

        # convert column to a (possibly very) long string
        long_string = column_data.to_string(index=False, header=False)
        # perform all processing in lower-case
        long_string = long_string.lower()
        # replace nan and frieds by missing value keyword
        long_string = multiple_replace(
            {
                'nan': '__missing__',
                'np.nan': '__missing__',
                'nat': '__missing__',
                'np.nat': '__missing__',
                'null': '__missing__',
                'none': '__missing__',
            }, long_string)
        # replace missing keyword by a space
        long_string = long_string.replace('__missing__', ' ')
        # replace new line escape characters by spaces
        long_string = multiple_replace({
                '\n': ' ',
                '\r': ' ',
                '\t': ' ',
            }, long_string)
        # replace multiple spaces by single space
        long_string = re.sub(' +', ' ', long_string)

        # iterate over wha'ts left to create sentences
        for c in long_string:
            # add one character at a time
            sentence = sentence + c

            # trigger sentence packing
            # - when sentence is about to get longer than what
            #   the tokenizer allows without truncation
            if len(sentence) > self.max_length - 1:
                # avoid abrupt truncation of sentence
                cc = c
                while cc not in [' ', '.'] and len(sentence) > 1:
                    sentence = sentence[:-1]
                    cc = sentence[-1]
                ready = True
            # - when reaching the end of the column (for short tables)
            elif sentence == long_string:
                ready = True

            # pack sentence
            if ready:
                # add sentence to list of sentences
                sentences.append(sentence)
                # reset sentence buffer
                sentence = ''
                # reset ready flag
                ready = False

        return sentences

    def add_training_column(self,
                            column_data: pandas.Series,
                            column_type: str):
        """ Adds column data for training.
        """
        # replace missing values that pandas can recognize
        # by a keyword ('__missing__')
        column_data = column_data.fillna('__missing__')
        sentences = self.column_to_sentences(column_data)
        labels = numpy.asarray([column_type for _ in sentences]).ravel()
        labels = self.type_encoder.transform(labels)
        labels = labels.ravel().tolist()

        self.sentence_list += sentences
        self.label_list += labels

    def prepare_dataset(self):
        """ Converts sentence-label pairs into dataset and dataloaders.
        """
        # ensure label_list has even number of elements
        if len(self.label_list) % 2 == 1:
            self.label_list.pop()

        # stratified splitting
        ids = numpy.arange(len(self.label_list)).tolist()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        sss.get_n_splits(ids, self.label_list)
        train_ids, val_ids = next(sss.split(ids, self.label_list))

        # ensure even val and train id arrays
        if len(val_ids) % 2 != 0:
            val_ids = val_ids[:-1]
        if len(train_ids) % 2 != 0:
            train_ids = train_ids[:-1]

        # build train/val sets
        val_sentences_i = []
        val_sentences_j = []
        scores_ij = []
        train_examples = []
        train_examples_2 = []
        for pos in range(len(self.sentence_list))[:-1:2]:
            # add training example
            lbl_1 = self.label_list[pos]
            lbl_2 = self.label_list[pos + 1]
            sen_1 = self.sentence_list[pos]
            sen_2 = self.sentence_list[pos + 1]
            lbl = 1.0
            if lbl_1 != lbl_2:
                lbl = 0.0
            if pos in train_ids:
                ex = InputExample(texts=[sen_1, sen_2], label=lbl)
                ex2_1 = InputExample(texts=[sen_1], label=lbl_1)
                ex2_2 = InputExample(texts=[sen_2], label=lbl_2)
                train_examples.append(ex)
                train_examples_2.append(ex2_1)
                train_examples_2.append(ex2_2)
            # add to validation examples
            elif pos in val_ids:
                val_sentences_i.append(sen_1)
                val_sentences_j.append(sen_2)
                scores_ij.append(lbl)

        train_examples = SentencesDataset(train_examples, self.model)
        train_examples_2 = SentencesDataset(train_examples_2, self.model)
        self.evaluator = EmbeddingSimilarityEvaluator(
            val_sentences_i,
            val_sentences_j,
            scores_ij,
            write_csv='eval_results.csv',
            show_progress_bar=True
        )

        self.train_dataloader = DataLoader(train_examples,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=2,
                                           persistent_workers=True)
        self.train_dataloader_2 = DataLoader(train_examples_2,
                                             shuffle=True,
                                             batch_size=self.batch_size,
                                             num_workers=2,
                                             persistent_workers=True)

    def finetune(self, n_epochs: int = 1, n_warmup: int = 1):
        """ Fine tunes BERT for sentence classification.
        """
        self.model.to(self.device)
        train_loss = OnlineContrastiveLoss(model=self.model)
        train_loss_2 = BatchHardSoftMarginTripletLoss(model=self.model)
        self.model.fit(train_objectives=[
                           (self.train_dataloader, train_loss),
                           (self.train_dataloader_2, train_loss_2)],
                       epochs=n_epochs,
                       evaluator=self.evaluator,
                       evaluation_steps=50,
                       optimizer_params={'lr': 1e-5},
                       weight_decay=1e-6,
                       show_progress_bar=True,
                       save_best_model=True,
                       output_path='./current_model/',
                       checkpoint_save_steps=50,
                       checkpoint_path='./current_model/checkpoints')

        print("final score:", self.model.evaluate(self.evaluator))

    def save(self, model_path: str):
        """ save model.
        """
        # Saving best-practices: if you use defaults names for the model
        # you can reload it using from_pretrained()
        output_dir = model_path
        # create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Saving model to {output_dir}")
        # take care of case of parallel training
        model_to_save = self.model.module \
            if hasattr(self.model, 'module') else self.model
        model_to_save.save(output_dir)

    def load(self, model_path: str):
        """ Load model from disk.
        """
        self.model = SentenceTransformer(
            model_path,
            device=self.device)

    def get_column_embeddings(self, column_data: pandas.Series) -> list:
        """ Returns the data type of the column.
        """
        self.model.eval()
        sentences = self.column_to_sentences(column_data)
        embeddings = self.model.encode(sentences)
        embeddings = numpy.atleast_2d(embeddings)

        return embeddings
