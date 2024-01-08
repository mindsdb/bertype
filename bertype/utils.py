import os
import glob
import pandas

from bertype.column_info import get_types
from bertype.column_info import get_subtypes
from bertype.column_info import get_modifiers


def load_data_and_annotations(data_path: str,
                              max_n_rows: int = 10000,
                              raise_on_error: bool = True):
    """ Utilitary function to load data and annotations.
    """
    data = []
    types = []
    subtypes = []
    modifiers = []
    files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    data_files = [name for name in files if not name.endswith('_annot.csv')]
    annt_files = [name for name in files if name.endswith('_annot.csv')]
    for data_file, annot_file in zip(data_files, annt_files):
        an = pandas.read_csv(annot_file, index_col=0)
        df = pandas.read_csv(data_file, nrows=max_n_rows, low_memory=False)
        print(data_file, annot_file)
        for col_name, col_info in an.iterrows():
            # safe check type
            if 'column_type' not in col_info:
                raise ValueError(f'[CRITICAL] file {annot_file} invalid type for entry {col_name}')
            col_type = col_info['column_type']
            # safe check subtype
            if 'column_subtype' not in col_info:
                raise ValueError(f'[CRITICAL] file {annot_file} invalid subtype for entry {col_name}')
            col_subtype = col_info['column_subtype']

            if 'column_modifier' not in col_info:
                raise ValueError(f'[CRITICAL] file {annot_file} invalid modifier for entry {col_name}')
            col_modifier = col_info['column_modifier']
            # safety checks
            # - column is labeled
            if pandas.isna(col_type):
                print(f"[ERROR] Annotation file {annot_file} contains invalid data for entry {col_name}")
                if raise_on_error:
                    raise ValueError("[ERROR] Cannot load annotations.")
            # column type is valid
            col_type = str(col_type)
            if col_type not in get_types():
                print(f"[ERROR] Annotation file {annot_file} contains invalid type for entry {col_name}")
                if raise_on_error:
                    raise ValueError("[ERROR] Cannot load annotations.")
                else:
                    continue
            # column subtype is valid
            if col_subtype not in get_subtypes():
                print(f"[ERROR] Annotation file {annot_file} contains invalid subtype for entry {col_name}")
                if raise_on_error:
                    raise ValueError("[ERROR] Cannot load annotations.")
                else:
                    continue
            # column modifier is valid
            if col_modifier not in get_modifiers():
                print(f"[ERROR] Annotation file {annot_file} contains invalid modifier for entry {col_name}")
                if raise_on_error:
                    raise ValueError("[ERROR] Cannot load annotations.")
                else:
                    continue
            # only then add data.
            col_data = df[col_name].copy()
            data.append(col_data)
            types.append(col_type)
            subtypes.append(col_subtype)
            modifiers.append(col_modifier)

    if not (len(data) == len(types) == len(subtypes) == len(modifiers)):
        print("[ERROR] Inconsistent data lengths. Cannot continue.")
        raise RuntimeError("[CRITICAL] Processing error.")

    r = {
        'data': data,
        'types': types,
        'subtypes': subtypes,
        'modifiers': modifiers
    }
    return r