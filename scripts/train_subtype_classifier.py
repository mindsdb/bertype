import sys
import numpy
from tqdm import tqdm

from bertype.embedders import ColBert
from bertype.classifiers import SimpleSubTypeClassifier
from bertype.utils import load_data_and_annotations


if __name__ == '__main__':

    checkpoint_path = sys.argv[1]
    embedder = ColBert(
        model_name=checkpoint_path,
        device='cuda'
    )

    classifier = SimpleSubTypeClassifier(
        embedder.get_embedding_length(),
        device='cuda'
    )
    # load data
    packed = load_data_and_annotations('./data/annotated')
    data = packed['data']
    subtypes = packed['subtypes']

    # transform type names into type ordinals (0, 1 or 2)
    type_codes = embedder.subtype_encoder.transform(subtypes)

    # compute embeddings to train classifier
    embeddings = []
    column_subtypes = []
    for col_data, col_subtype in tqdm(zip(data, type_codes), total=len(data)):
        embs = embedder.get_column_embeddings(col_data)
        if embs.shape[1] < 1:
            print(embs.shape)
            print("invalid embedding found.")
        else:
            embeddings.append(embs)
            column_subtypes.append(numpy.repeat(col_subtype, len(embs)))
    embeddings = numpy.concatenate(embeddings, axis=0)
    column_subtypes = numpy.concatenate(column_subtypes, axis=0)

    # counts is sorted in descending order
    _, counts = numpy.unique(column_subtypes, return_counts=True)
    weights = None  # 1.0 - counts / numpy.sum(counts)
    # resampler = SMOTEENN(sampling_strategy=)
    # embeddings, type_codes = resampler.fit_resample(embeddings, type_codes)

    _, counts = numpy.unique(column_subtypes, return_counts=True)
    print(counts)

    classifier.fit(embeddings, column_subtypes,
                   n_epochs=50,
                   weights=weights,
                   evaluate_every=5,
                   checkpoint_every=5)
    classifier.save('subtype_classifier.pth')
