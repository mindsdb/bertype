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
from bertype.column_info import get_type_codes
from bertype.column_info import get_subtypes
from bertype.column_info import get_subtype_codes
from bertype.column_info import decode_types
from bertype.column_info import decode_subtypes


class Simple:
    """ Colbert + SimpleClassifier implementation.
    """
    __N_TYPES__ = len(get_types())

    def __init__(self):
        """ Initializer.
        """
        self.embedder_ = torch.nn.Module
        self.type_clf_ = torch.nn.Module
        self.subtype_clf_models_ = {}

        self.pval_thresh_ = 0.01

    def attach_embedder(self, emb: torch.nn.Module):
        """ Adds embedder to engine.
        """
        self.embedder_ = emb

    def attach_type_classifier(self, clf: torch.nn.Module):
        """ Adds global type classifier to engine.
        """
        self.type_clf_ = clf

    def attach_subtype_classifier(self, parent_type: str, clf: torch.nn.Module):
        """ Adds subtype classifier for given type.

            :param parent_type (str)
                'text', 'number' or 'timestamp'.
            :param clf (torch.nn.Module)
                a PyTorch nn.Module (model)

            :raises ValueError
                if `parent_type` is not valid (see get_types())
        """
        if parent_type not in get_types():
            raise ValueError(f'[ERROR] uknown type {parent_type}')
        if parent_type in self.subtype_clf_models_:
            print(f'[WARNING] Overwriting model for type {parent_type}')
        self.subtype_clf_models_[parent_type] = clf

    def infer_column_subtype(self, column_type: str,
                             column_embeddings: numpy.ndarray):
        """ Returns likely column subtype.
        """
        # get data types for of each embedding
        x = torch.tensor(column_embeddings, dtype=torch.float32)
        with torch.no_grad():
            probs = self.subtype_clf_models_[column_type](x)
        # obtain type code (ordinal representation of type)
        # for all embeddings and maka them numpy arrays
        type_codes = torch.argmax(probs, dim=-1).cpu().numpy().ravel()
        # calculate frequencies (counts) of each type code given
        # the embeddings. This tiny loop is needed to take into
        # account the gaps
        likely_types, freqs = numpy.unique(type_codes, return_counts=True)
        type_freqs = numpy.zeros(len(get_subtypes(column_type)))
        for i, t in enumerate(likely_types):
            type_freqs[t] = freqs[i]
        likely_type_code = numpy.argmax(type_freqs)
        # use chi-square test to obtain an estimation of how
        # likely it is for the column to belong to a certain
        # data-type. The null hypotesis is that the type of a
        # column is a random variable with a uniform probability
        # distribution. This only works well for tables with
        # at least 100-ish entries, which is way below the target
        # of this implementation (millions of rows)
        data_stats = chisquare(type_freqs)
        # find the frequencies at which the p-value becomes small
        # enough for the measurement to be statistically significant
        # compared what has been measured. Yes, this is sort of a
        # reverse Chi-Square test, absolutely noqa
        p = 1.0
        direction = 1
        f = numpy.floor(1.0 / len(get_subtypes(column_type)) * len(column_embeddings))
        r = len(column_embeddings) - len(get_subtypes(column_type)) * f
        null_freqs = numpy.repeat(f, len(get_subtypes(column_type))).astype(int)
        null_freqs[0] += r
        old_freqs = numpy.copy(null_freqs)
        new_freqs = numpy.copy(old_freqs)

        iter = 0
        max_iter = 100
        while iter < max_iter and p > 0.001:
            # max value if jumping forwards is the sum of all counts
            # that are not the likely type
            max_forward = numpy.sum(old_freqs) - old_freqs[likely_type_code]
            if max_forward <= 1:
                direction = -direction
                iter = iter + 1
                continue
            # max value if jumping backwards is the number of counts
            max_backwards = old_freqs[likely_type_code]
            if max_backwards <= 1:
                direction = -direction
                iter = iter + 1
                continue
            f0 = 0
            # jump forward
            if direction == 1:
                f0 = numpy.random.randint(1, max_forward)
            # jump backwards
            elif direction == -1:
                f0 = numpy.random.randint(1, max_backwards)
            n0 = old_freqs[likely_type_code] + direction * f0
            # update likely type
            new_freqs[likely_type_code] = n0
            # update the rest
            updated = [likely_type_code, ]
            while len(updated) != len(get_subtypes(column_type)):
                tc = numpy.random.choice(get_subtype_codes(column_type))
                if tc in updated:
                    continue
                # if sampling the last one, subtract what's left
                if len(updated) + 1 == len(get_subtypes(column_type)):
                    new_freqs[tc] = old_freqs[tc] - direction * f0
                    updated.append(tc)
                    continue
                nc = 1
                if f0 > 1:
                    nc = numpy.random.randint(1, f0)
                # if nc is too large, sample as much as possible
                if old_freqs[tc] - direction * nc <= 0:
                    nc = old_freqs[tc]
                new_freqs[tc] = old_freqs[tc] - direction * nc
                f0 = f0 - nc
                updated.append(tc)
            # chi-square stats of dummy frequencies
            stats = chisquare(new_freqs, null_freqs)
            p = stats.pvalue
            print(old_freqs)
            print(new_freqs)
            print(f'{p:2.6f} {data_stats.pvalue:2.6e}')
            # update old frequencies
            old_freqs = numpy.copy(new_freqs)

            iter = iter + 1

        # compare normalized frequencies for data and "simulation"
        print('-------------------------------------')
        print('< BEGIN REPORT >')
        print(f'Likely type name: {decode_subtypes(column_type, [likely_type_code,])[0]}')
        print(f'p-value of data: {data_stats.pvalue:+04.4e}')
        print(f'p-value of simulation: {p:+04.4e}')
        print('TABLE WITH RELATIVE FREQUENCIES')
        print(' data simulation')
        for d, s in zip(type_freqs, old_freqs.astype(int)):
            di = int(d)
            si = int(s)
            print(f'{di:+04d}  {si:+04d}')
        print('< END REPORT >')
        print('-------------------------------------')

        subtype_name = decode_subtypes(column_type, [likely_type_code, ])[0]

        return subtype_name

    def infer_column_type(self, column_embeddings: numpy.ndarray):
        """ Returns the most likely type given embeddings.
        """
        # get data types for of each embedding
        x = torch.tensor(column_embeddings, dtype=torch.float32)
        with torch.no_grad():
            probs = self.type_clf_(x)
        # obtain type code (ordinal representation of type)
        # for all embeddings and maka them numpy arrays
        type_codes = torch.argmax(probs, dim=-1).cpu().numpy().ravel()
        # calculate frequencies (counts) of each type code given
        # the embeddings. This tiny loop is needed to take into
        # account the gaps
        likely_types, freqs = numpy.unique(type_codes, return_counts=True)
        type_freqs = numpy.zeros(len(get_types()))
        for i, t in enumerate(likely_types):
            type_freqs[t] = freqs[i]
        likely_type_code = numpy.argmax(type_freqs)
        # use chi-square test to obtain an estimation of how
        # likely it is for the column to belong to a certain
        # data-type. The null hypotesis is that the type of a
        # column is a random variable with a uniform probability
        # distribution. This only works well for tables with
        # at least 100-ish entries, which is way below the target
        # of this implementation (millions of rows)
        data_stats = chisquare(type_freqs)
        # find the frequencies at which the p-value becomes small
        # enough for the measurement to be statistically significant
        # compared what has been measured. Yes, this is sort of a
        # reverse Chi-Square test, absolutely noqa
        p = 1.0
        direction = 1
        f = numpy.floor(1.0 / len(get_types()) * len(column_embeddings))
        r = len(column_embeddings) - len(get_types()) * f
        null_freqs = numpy.repeat(f, len(get_types())).astype(int)
        null_freqs[0] += r
        old_freqs = numpy.copy(null_freqs)
        new_freqs = numpy.copy(old_freqs)

        iter = 0
        max_iter = 100
        while iter < max_iter and p > 0.001:
            # max value if jumping forwards is the sum of all counts
            # that are not the likely type
            max_forward = numpy.sum(old_freqs) - old_freqs[likely_type_code]
            if max_forward <= 1:
                direction = -direction
                iter = iter + 1
                continue
            # max value if jumping backwards is the number of counts
            max_backwards = old_freqs[likely_type_code]
            if max_backwards <= 1:
                direction = -direction
                iter = iter + 1
                continue
            f0 = 0
            # jump forward
            if direction == 1:
                f0 = numpy.random.randint(1, max_forward)
            # jump backwards
            elif direction == -1:
                f0 = numpy.random.randint(1, max_backwards)
            n0 = old_freqs[likely_type_code] + direction * f0
            # update likely type
            new_freqs[likely_type_code] = n0
            # update the rest
            updated = [likely_type_code, ]
            while len(updated) != len(get_types()):
                tc = numpy.random.choice(get_type_codes())
                if tc in updated:
                    continue
                # if sampling the last one, subtract what's left
                if len(updated) + 1 == len(get_types()):
                    new_freqs[tc] = old_freqs[tc] - direction * f0
                    updated.append(tc)
                    continue
                nc = 1
                if f0 > 1:
                    nc = numpy.random.randint(1, f0)
                # if nc is too large, sample as much as possible
                if old_freqs[tc] - direction * nc <= 0:
                    nc = old_freqs[tc]
                new_freqs[tc] = old_freqs[tc] - direction * nc
                f0 = f0 - nc
                updated.append(tc)
            # chi-square stats of dummy frequencies
            stats = chisquare(new_freqs, null_freqs)
            p = stats.pvalue
            print(old_freqs)
            print(new_freqs)
            print(f'{p:2.6f} {data_stats.pvalue:2.6e}')
            # update old frequencies
            old_freqs = numpy.copy(new_freqs)

            iter = iter + 1

        # compare normalized frequencies for data and "simulation"
        print('-------------------------------------')
        print('< BEGIN REPORT >')
        print(f'Likely type name: {decode_types([likely_type_code,])[0]}')
        print(f'p-value of data: {data_stats.pvalue:+04.4e}')
        print(f'p-value of simulation: {p:+04.4e}')
        print('TABLE WITH RELATIVE FREQUENCIES')
        print(' data simulation')
        for d, s in zip(type_freqs, old_freqs.astype(int)):
            di = int(d)
            si = int(s)
            print(f'{di:+04d}  {si:+04d}')
        print('< END REPORT >')
        print('-------------------------------------')

        type_name = decode_types([likely_type_code, ])[0]

        return type_name

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
            col_type_name = self.infer_column_type(col_embs)
            col_subtype_name = self.infer_column_subtype(col_type_name, col_embs)
            # # get data sub-types for of each embedding
            # probs = torch.zeros((len(col_embs), len(get_subtypes())))
            # with torch.no_grad():
            #     for i, emb in enumerate(col_embs):
            #         x = torch.tensor(emb, dtype=torch.float32)
            #         probs[i] = self.subtype_clf_(x)
            # subtypes = torch.argmax(probs, dim=-1).cpu().numpy().ravel()
            # # noqa
            # # if table is small, metric is set to a *very rough*
            # # estimation of the uncertainty in the probability
            # # of the column corresponding to a given type.
            # # otherwise, use chi-square test
            # likely_subtypes, subfreqs = numpy.unique(subtypes, return_counts=True)
            # global_subprobs = subfreqs / numpy.sum(subfreqs)
            # subtype_freqs = numpy.zeros(len(get_subtypes()))
            # subtype_probs = numpy.zeros(len(get_subtypes()))
            # for i, t in enumerate(likely_subtypes):
            #     subtype_freqs[t] = subfreqs[i]
            #     subtype_probs[t] = global_subprobs[i]
            # subtype_code = numpy.argmax(subtype_probs)
            # col_subtype = self.embedder_.decode_subtype(subtype_code)
            # # test results against null hypotesis of all types
            # # being equally likely to be (noqa)
            # res2 = chisquare(subtype_freqs)
            # print('# BEGIN INFERENCE DIAGNOSIS INFO BLOCK ')
            # print(f'  column name: {col_name}')
            # print(f'  column length: {len(col_data)}')
            # print(f'  p-value for type {col_type}: {res.pvalue:1.5f}')
            # print(f'  p-value for sub-type {col_subtype}: {res2.pvalue:1.5f}')
            # print('')

            column_types[col_name] = col_type_name
            column_subtypes[col_name] = col_subtype_name

        return column_types, column_subtypes
