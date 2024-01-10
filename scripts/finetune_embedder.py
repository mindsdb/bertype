""" finetune_embedder.py

    Fine-tunes sentence transformer to distinguish between column types.
"""
from tqdm import tqdm

from bertype.embedders import ColBert
from bertype.column_info import encode_types
from bertype.utils import load_data_and_annotations


if __name__ == '__main__':

    # prepare embedder
    embedder = ColBert(device='cuda:0')
    # load data
    packed = load_data_and_annotations('./data/')
    data = packed['data']
    types = packed['types']
    # transform type names into type ordinals (0, 1 or 2)
    type_codes =  encode_types(types)

    # attach data
    for d, t in tqdm(zip(data, types), total=len(data)):
        embedder.add_training_column(d, t)
    # prepare dataset/dataloaders
    embedder.prepare_dataset()
    embedder.finetune(n_epochs=100)
