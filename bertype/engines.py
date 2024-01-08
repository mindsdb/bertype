""" engines.py

Implementation of different column type classifier engines
that use different combinations of embedders and classification
methods.

Currrently only ColBert + Simple{Type, Mod}Classifier are supported.
"""
import torch
import numpy
import pandas

from scipy.stats import chisquare

from bertype.column_info import get_types
from bertype.column_info import get_subtypes


class Simple:
    """ Colbert + SimpleClassifier implementation.
    """
    __N_TYPES__ = len(get_types())

    def __init__(self, embedder, type_clf, subtype_clf):
        """ Initializer.
        """
        self.embedder_ = embedder
        self.type_clf_ = type_clf
        self.subtype_clf_ = subtype_clf

    def infer(self, dataframe: pandas.DataFrame) -> dict:
        """ Performs inference of column types.
        """
        self.type_clf_.eval()
        column_types = {}
        column_subtypes = {}
        for col_name in dataframe.columns.to_list():
            # get column data
            col_data = dataframe[col_name].copy()

            # get column embeddings
            col_embs = self.embedder_.get_column_embeddings(col_data)

            # get data types for of each embedding
            probs = torch.zeros((len(col_embs), len(get_types())))
            with torch.no_grad():
                for i, emb in enumerate(col_embs):
                    x = torch.tensor(emb, dtype=torch.float32)
                    probs[i] = self.type_clf_(x)
            types = torch.argmax(probs, dim=-1).cpu().numpy().ravel()
            # noqa
            # if table is small, metric is set to a *very rough*
            # estimation of the uncertainty in the probability
            # of the column corresponding to a given type.
            # otherwise, use chi-square test
            likely_types, freqs = numpy.unique(types, return_counts=True)
            global_probs = freqs / numpy.sum(freqs)
            type_freqs = numpy.zeros(len(get_types()))
            type_probs = numpy.zeros(len(get_types()))
            for i, t in enumerate(likely_types):
                type_freqs[t] = freqs[i]
                type_probs[t] = global_probs[i]
            type_code = numpy.argmax(type_probs)
            col_type = self.embedder_.decode_type(type_code)
            # test results against null hypotesis of all types
            # being equally likely to be (noqa)
            res = chisquare(type_freqs)

            # get data sub-types for of each embedding
            probs = torch.zeros((len(col_embs), len(get_subtypes())))
            with torch.no_grad():
                for i, emb in enumerate(col_embs):
                    x = torch.tensor(emb, dtype=torch.float32)
                    probs[i] = self.subtype_clf_(x)
            subtypes = torch.argmax(probs, dim=-1).cpu().numpy().ravel()
            # noqa
            # if table is small, metric is set to a *very rough*
            # estimation of the uncertainty in the probability
            # of the column corresponding to a given type.
            # otherwise, use chi-square test
            likely_subtypes, subfreqs = numpy.unique(subtypes, return_counts=True)
            global_subprobs = subfreqs / numpy.sum(subfreqs)
            subtype_freqs = numpy.zeros(len(get_subtypes()))
            subtype_probs = numpy.zeros(len(get_subtypes()))
            for i, t in enumerate(likely_subtypes):
                subtype_freqs[t] = subfreqs[i]
                subtype_probs[t] = global_subprobs[i]
            subtype_code = numpy.argmax(subtype_probs)
            col_subtype = self.embedder_.decode_subtype(subtype_code)
            # test results against null hypotesis of all types
            # being equally likely to be (noqa)
            res2 = chisquare(subtype_freqs)
            print('# BEGIN INFERENCE DIAGNOSIS INFO BLOCK ')
            print(f'  column name: {col_name}')
            print(f'  column length: {len(col_data)}')
            print(f'  p-value for type {col_type}: {res.pvalue:1.5f}')
            print(f'  p-value for sub-type {col_subtype}: {res2.pvalue:1.5f}')
            print('')

            column_types[col_name] = col_type
            column_subtypes[col_name] = col_subtype

        return column_types, column_subtypes
