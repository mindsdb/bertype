"""
    Evaluate embeddings by plotting them in 2-D space.
"""
import os
import sys
from tqdm import tqdm

import numpy

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from bertype.embedders import ColBert
from bertype.utils import load_data_and_annotations


if __name__ == '__main__':

    run_folder = sys.argv[1]
    checkpoints = sys.argv[2::]
    checkpoints_sorted = sorted(checkpoints,
                                key=lambda x: int(os.path.basename(x)))
    # load data
    packed = load_data_and_annotations('./data/')
    data = packed['data']
    types = packed['types']

    for model_chk in checkpoints_sorted:

        embedder = ColBert('cuda', model_name=model_chk)
        # convert type names to ordinals
        type_codes = embedder.type_encoder.transform(types)
        # calculate embeddings
        embeddings = []
        expanded_type_codes = []
        for col_data, type_code in tqdm(zip(data, type_codes), total=len(data)):
            embs = embedder.get_column_embeddings(col_data)
            for em in embs:
                embeddings.append(em)
                expanded_type_codes.append(type_code)
        expanded_type_codes = numpy.asarray(expanded_type_codes)

        # build plot to check for auto-magic clustering
        pca = PCA(n_components=2)
        embeddings -= numpy.average(embeddings, axis=0)
        xy = pca.fit_transform(embeddings)
        x, y = xy[:, 0], xy[:, 1]

        print('plotting')
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.set_title(f'model_chk (checkpoint) {os.path.basename(model_chk)}')

        colors = ['red', 'green', 'blue']
        for t in [0, 1, 2]:
            mask = expanded_type_codes == t
            ax.scatter(x[mask], y[mask],
                       s=2.0,
                       alpha=0.3,
                       c=colors[t])
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-10.5, 10.5)
        ax.set_aspect('equal')
        fig.tight_layout()

        plot_path = os.path.join(
            run_folder,
            f'embeddings_chk_{os.path.basename(model_chk)}.png'
        )
        print(f'saving plot for model for {model_chk} at {plot_path}')
        plt.savefig(plot_path)
