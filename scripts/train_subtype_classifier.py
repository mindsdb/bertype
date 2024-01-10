import sys
import numpy
from tqdm import tqdm

from bertype.embedders import ColBert
from bertype.classifiers import SimpleSubTypeClassifier
from bertype.column_info import get_types
from bertype.column_info import get_subtypes
from bertype.column_info import encode_types, decode_types
from bertype.column_info import encode_subtypes
from bertype.utils import load_data_and_annotations


if __name__ == '__main__':

    checkpoint_path = sys.argv[1]
    embedder = ColBert(
        model_name=checkpoint_path,
        device='cuda'
    )
    # load data
    packed = load_data_and_annotations('./data/annotated')
    data = packed['data']
    types = packed['types']
    subtypes = packed['subtypes']

    # compute embeddings to train classifier
    column_embeddings = []
    column_type_codes = []
    column_subtype_codes = {}
    for t in get_types():
        column_subtype_codes[t] = []

    for col_data, col_type, col_subtype in tqdm(
          zip(data, types, subtypes), total=len(data)):
        # get embedding of column
        embs = embedder.get_column_embeddings(col_data)
        # check embedding are valid
        if embs.shape[1] < 1:
            print(embs.shape)
            print("invalid embedding found.")
        else:
            column_embeddings.append(embs)
            column_type_codes.append(
                numpy.repeat(
                    encode_types([col_type,]),
                    len(embs)
                )
            )
            column_subtype_codes[col_type].append(
                numpy.repeat(
                    encode_subtypes(col_type, [col_subtype,]),
                    len(embs)
                )
            )

    # convert lists to numpy array
    embeddings = numpy.concatenate(column_embeddings, axis=0)
    column_type_codes = numpy.concatenate(column_type_codes, axis=0)
    # train sub-type classifiers
    for t in get_types():
        print(f'training classifier for type {t}')
        t_code = encode_types([t,])
        type_mask = column_type_codes == t_code
        st = numpy.concatenate(column_subtype_codes[t])
        em = embeddings[type_mask]
        print(em.shape, st.shape)

        classifier = SimpleSubTypeClassifier(
            t,
            embedder.get_embedding_length(),
            device='cuda'
        )
        classifier.fit(em, st,
                       n_epochs=50,
                       evaluate_every=5,
                       checkpoint_every=5)
        classifier.save(f'{t}_subtype_classifier.pth')
