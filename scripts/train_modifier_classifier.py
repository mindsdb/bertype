import sys
import numpy
from tqdm import tqdm

from bertype.embedders import ColBert
from bertype.classifiers import SimpleModifierClassifier
from bertype.utils import load_data_and_annotations
from bertype.column_info import encode_modifiers


if __name__ == '__main__':

    checkpoint_path = sys.argv[1]
    embedder = ColBert(
        model_name=checkpoint_path,
        device='cuda'
    )

    # load data
    packed = load_data_and_annotations('./data/annotated', max_n_rows=10000, max_n_datasets=10)
    data = packed['data']
    modifiers = packed['modifiers']
    index = packed['dataset_index']
    # transform type names into type ordinals (0, 1 or 2)
    modifier_codes = encode_modifiers(modifiers)

    # compute embeddings to train classifier and build
    # target column
    embeddings = []
    n_embs_per_col = []
    column_modifiers = []
    for col_data, col_modifier in \
            tqdm(zip(data, modifier_codes), total=len(data)):
        embs = embedder.get_column_embeddings(col_data)
        if embs.shape[1] < 1:
            print(embs.shape)
            print("invalid embedding found.")
        else:
            n_embs_per_col.append(len(embs))
            embeddings.append(embs)
            column_modifiers.append(col_modifier)
    column_modifiers = numpy.asarray(column_modifiers)

    # train data type classifier
    classifier = SimpleModifierClassifier(
        embedder.get_embedding_length(),
        device='cuda'
    )
    classifier.fit(embeddings, column_modifiers,
                   index, n_embs_per_col,
                   n_epochs=500,
                   evaluate_every=5,
                   checkpoint_every=5)
    classifier.save('modifier_classifier.ptk')
