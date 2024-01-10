import os
import glob
import numpy
import pandas

from bertype.column_info import get_types
from bertype.column_info import get_subtypes
from bertype.column_info import get_modifiers
from bertype.column_info import get_invalid_keys


def health_checks(col_name, col_info, raise_on_error: bool = False):
    """ Returns True if everything is OK.
    """
    # safe check type
    if 'column_type' not in col_info:
        if raise_on_error:
            raise ValueError(f'[CRITICAL] no type anotation for {col_name}')
        return False
    col_type = col_info['column_type']

    # safe check subtype
    if 'column_subtype' not in col_info:
        if raise_on_error:
            raise ValueError(f'[CRITICAL] no subtype annotation for {col_name}')
        return False
    col_subtype = col_info['column_subtype']

    # check modifier
    if 'column_modifier' not in col_info:
        if raise_on_error:
            raise ValueError(f'[CRITICAL] no modifier annotations for {col_name}')
        return False
    col_modifier = col_info['column_modifier']

    # safety checks
    # - column is labeled
    if pandas.isna(col_type):
        print(f"[ERROR] invalid column typa {col_type} for entry {col_name}")
        if raise_on_error:
            raise ValueError("[ERROR] Cannot load annotations.")
        return False
    # - column type is valid
    col_type = str(col_type)
    if col_type not in get_types():
        print(f"[ERROR] unknown column type {col_type} for entry {col_name}")
        if raise_on_error:
            raise ValueError("[ERROR] Cannot load annotations.")
        else:
            return False
    # - column subtype is valid
    if col_subtype not in get_subtypes(col_type):
        print(f"[ERROR] unknown column subtype {col_subtype} for entry {col_name}")
        if raise_on_error:
            raise ValueError("[ERROR] Cannot load annotations.")
        else:
            return False
    # - column modifier is valid
    if col_modifier not in get_modifiers():
        print(f"[ERROR] unknown modifier {col_modifier} for entry {col_name}")
        if raise_on_error:
            raise ValueError("[ERROR] Cannot load annotations.")
        else:
            return False

    return True


def load_data_and_annotations(data_path: str,
                              max_n_rows: int = 100000,
                              max_n_datasets: int = -1,
                              augment: bool = True,
                              raise_on_error: bool = False):
    """ Utilitary function to load data and annotations.
    """
    data = []
    types = []
    subtypes = []
    modifiers = []
    dataset_index = []

    files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    data_files = [name for name in files if not name.endswith('_annot.csv')]
    annt_files = [name for name in files if name.endswith('_annot.csv')]
    data_files = data_files[0:max_n_datasets]
    annt_files = annt_files[0:max_n_datasets]
    for idx, (data_file, annot_file) in enumerate(zip(data_files, annt_files)):
        an = pandas.read_csv(annot_file, index_col=0)
        df = pandas.read_csv(data_file, nrows=max_n_rows, low_memory=False)
        print(f'[INFO] loading data from {data_file} ({annot_file})')
        if augment:
            print('[INFO] Data augmentation is enabled.')
            df_aug = df.copy()
            df_aug.reset_index(inplace=True, drop=True)
            an_aug = an.copy()
            for col_name, col_info in an.iterrows():
                # check health. skip if bad
                if not health_checks(col_name, col_info):
                    continue
                # extract column modifier
                col_modifier = col_info['column_modifier']
                # only augment nominal columns
                if col_modifier != 'nominal':
                    continue
                # iterate over all modifiers
                for modifier in get_modifiers():
                    print(f'[INFO] Augmenting {col_name} to have modifier {modifier}.')
                    # do not augment self!
                    if modifier == col_modifier:
                        continue
                    x = None
                    aug_col_name = col_name + f'_augmented_{modifier}'
                    col_data = df[col_name].copy().astype(str)
                    # augment to make nominal column a constant column
                    # by randomly sampling an element and repeating it
                    if modifier == 'constant':
                        s = col_data.sample(1)
                        while s.values[0].lower() in get_invalid_keys():
                            s = col_data.sample(1)
                        x = numpy.repeat(s.values[0], len(col_data))
                    # augment to make categorical by randomly setting
                    # a number of categories and randomly pick them
                    # up to generate a len(data) vector
                    elif modifier == 'categorical':
                        categos = numpy.unique(col_data)
                        n_cat = numpy.random.randint(2, len(col_data) // 20)
                        n_cat = min(n_cat, len(categos))

                        c = []
                        for e in categos:
                            if e.lower() not in get_invalid_keys():
                                c.append(e)
                        if len(c) < 2:
                            print(f'[WARNING] column {col_name} marked as nominal but unique values are {categos}')
                            continue
                        x = numpy.random.choice(c, len(col_data))
                    # just in case
                    assert x is not None, \
                        "[FATALITY] This error means health-checks are useless."
                    new_annt = pandas.DataFrame(
                        index=[aug_col_name, ],
                        data=[{
                            'column_type': col_info['column_type'],
                            'column_subtype': col_info['column_subtype'],
                            'column_modifier': modifier
                        }]
                    )
                    new_entry = pandas.DataFrame(
                        index=numpy.arange(len(x)),
                        data=x,
                        columns=[aug_col_name,]
                    )
                    an_aug = pandas.concat([an_aug, new_annt])
                    df_aug = df_aug.join(new_entry, how='left')
            an = an_aug
            df = df_aug
            print(an.head())
            print(df.head())
        # count the number of valid columns
        n_cols = 0
        for col_name, col_info in an.iterrows():
            # check health. skip if bad
            if not health_checks(col_name, col_info):
                continue
            # only then add data.
            col_data = df[col_name].copy()
            data.append(col_data)
            types.append(col_info['column_type'])
            subtypes.append(col_info['column_subtype'])
            modifiers.append(col_info['column_modifier'])
        dataset_index.append(idx)

    if not (len(data) == len(types) == len(subtypes) == len(modifiers)):
        print("[ERROR] Inconsistent data lengths. Cannot continue.")
        raise RuntimeError("[CRITICAL] Processing error.")

    r = {
        'data': data,
        'types': types,
        'subtypes': subtypes,
        'modifiers': modifiers,
        'dataset_index': dataset_index,
    }

    return r

