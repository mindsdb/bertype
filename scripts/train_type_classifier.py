import sys
import numpy
from tqdm import tqdm

from bertype.embedders import ColBert
from bertype.classifiers import SimpleTypeClassifier
from bertype.utils import load_data_and_annotations


if __name__ == '__main__':

    checkpoint_path = sys.argv[1]
    embedder = ColBert(
        model_name=checkpoint_path,
        device='cuda'
    )

    classifier = SimpleTypeClassifier(
        embedder.get_embedding_length(),
        device='cuda'
    )
    # load data
    packed = load_data_and_annotations('./data/annotated')
    data = packed['data']
    types = packed['types']

    # transform type names into type ordinals (0, 1 or 2)
    type_codes = embedder.type_encoder.transform(types)

    # compute embeddings to train classifier
    embeddings = []
    column_types = []
    for col_data, col_type in tqdm(zip(data, type_codes), total=len(data)):
        embs = embedder.get_column_embeddings(col_data)
        if embs.shape[1] < 1:
            print(embs.shape)
            print("invalid embedding found.")
        else:
            embeddings.append(embs)
            column_types.append(numpy.repeat(col_type, len(embs)))
    embeddings = numpy.concatenate(embeddings, axis=0)
    column_types = numpy.concatenate(column_types, axis=0)

    # counts is sorted in descending order
    _, counts = numpy.unique(column_types, return_counts=True)
    weights = None  # 1.0 - counts / numpy.sum(counts)
    # resampler = SMOTEENN(sampling_strategy=)
    # embeddings, type_codes = resampler.fit_resample(embeddings, type_codes)

    _, counts = numpy.unique(column_types, return_counts=True)
    print(counts)

    classifier.fit(embeddings, column_types,
                   n_epochs=10,
                   weights=weights,
                   evaluate_every=1,
                   checkpoint_every=5)
    classifier.save('type_classifier.pth')
