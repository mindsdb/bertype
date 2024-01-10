import sys
import time
import pandas

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from type_infer.api import RuleBasedEngine

from bertype.engines import Simple
from bertype.embedders import ColBert
from bertype.classifiers import SimpleTypeClassifier
from bertype.classifiers import SimpleSubTypeClassifier
from bertype.column_info import get_types
from bertype.column_info import get_subtypes
from bertype.utils import load_data_and_annotations


# hack to use new API of type_infer
infer_types = RuleBasedEngine().infer


if __name__ == '__main__':

    embedder = ColBert(device='cuda:0')
    embedder.load(sys.argv[1])
    # e1 = embedder.model.encode(['this is a silly sentence'])
    # e2 = embedder.model.encode(['this is a silly sentence'])
    # print(e1[0][0:10])
    # print(e2[0][0:10])
    type_clf = SimpleTypeClassifier(
        embedder.get_embedding_length(),
        device='cuda:0'
    )
    type_clf.load(sys.argv[2])

    text_subtype_clf = SimpleSubTypeClassifier(
        'text',
        embedder.get_embedding_length(),
        device='cuda'
    )
    text_subtype_clf.load(sys.argv[3])

    number_subtype_clf = SimpleSubTypeClassifier(
        'number',
        embedder.get_embedding_length(),
        device='cuda'
    )
    number_subtype_clf.load(sys.argv[4])

    timestamp_subtype_clf = SimpleSubTypeClassifier(
        'timestamp',
        embedder.get_embedding_length(),
        device='cuda'
    )
    timestamp_subtype_clf.load(sys.argv[5])

    engine = Simple()
    engine.attach_embedder(embedder)
    engine.attach_type_classifier(type_clf)
    engine.attach_subtype_classifier('text', text_subtype_clf)
    engine.attach_subtype_classifier('number', number_subtype_clf)
    engine.attach_subtype_classifier('timestamp', timestamp_subtype_clf)

    gt_type = []
    bt_type = []
    ti_type = []

    # timing
    tinfer_times = {}
    betype_times = {}
    for t in get_types():
        for s in get_subtypes(t):
            tinfer_times[s] = 0.0
            betype_times[s] = 0.0

    # load data files
    packed = load_data_and_annotations('./data/annotated')
    data = packed['data']
    types = packed['types']
    subtypes = packed['subtypes']

    subtype_names = []
    for t in get_types():
        for s in get_subtypes(t):
            subtype_names.append(s)
    N = 100
    for col_data, col_type, col_subtype in zip(data[0:N], types[0:N], subtypes[0:N]):
        # transform to data-frame
        df = pandas.DataFrame(col_data)
        # get types from type_infer engine
        col_name = col_data.name
        tic1 = time.time()
        betype_type_info, betype_subtype_info = engine.infer(df)
        tic2 = time.time()
        # get types from bertype engine
        tinfer_info = infer_types(df).dtypes
        tic3 = time.time()
        # adjust type-infer predictions
        ignore = False
        tinfer_pred = tinfer_info[col_name]
        betype_pred = betype_subtype_info[col_name]
        if tinfer_pred == 'short_text':
            tinfer_pred = 'simple_text'
        elif tinfer_pred == 'rich_text':
            tinfer_pred = 'complex_text'
        elif tinfer_pred == 'quantity':
            tinfer_pred = betype_subtype_info[col_name]
        # if type-infer is binary or categorical, typeinfer does
        # not provide more information about the type
        # to be on the safe side, we consider the best case scenario
        # type_infer_pred = annotation
        elif tinfer_pred in ['binary', 'categorical', 'tags']:
            # ignore = True
            tinfer_pred = betype_subtype_info[col_name]
        # other types are ignored
        if tinfer_pred not in subtype_names:
            cst = col_subtype
            print(f'[WARNING] Uknown type {tinfer_pred} (annotated as {cst})')
            ignore = True

        if not ignore:
            gt_type.append(col_subtype)
            ti_type.append(tinfer_pred)
            bt_type.append(betype_pred)

            betype_times[tinfer_pred] += tic2 - tic1
            tinfer_times[betype_pred] += tic3 - tic2

    print('----------------------------------------')
    print("[TIMING REPORT FOR TYPE_INFER]")
    print(f'  processing rate: { N / sum(list(tinfer_times.values())):2.2f} col/sec')
    print('  disaggregated by types')
    for t in get_types():
        for st in get_subtypes(t):
            n = len([x for x in gt_type if x == st])
            if n == 0:
                continue
            print(f'    col_subtype: {st}')
            print(f'    rate: {(n / tinfer_times[st]):2.2f} sec / col')
            print('')
    print('----------------------------------------')

    print('')

    print('----------------------------------------')
    print("[TIMING REPORT FOR BERT BASED ENGINE]")
    print(f'  processing rate: {N / (sum(list(betype_times.values()))):2.2f} col/sec')
    print('  disaggregated by types')
    for t in get_types():
        for st in get_subtypes(t):
            n = len([x for x in gt_type if x == st])
            if n == 0:
                continue
            print(f'    col_subtype: {st}')
            print(f'    rate: {(n / betype_times[st]):2.2f} sec / col')
            print('')
    print('----------------------------------------')

    cm_gt_ti = confusion_matrix(gt_type, ti_type,
                                labels=subtype_names,
                                normalize='true')
    disp1 = ConfusionMatrixDisplay(cm_gt_ti,
                                   display_labels=subtype_names)
    disp1.plot(xticks_rotation='vertical')
    disp1.ax_.set_title('Type infer confusion matrix.')
    disp1.ax_.set_aspect('equal')
    figure = plt.gcf()
    figure.set_size_inches((7.5, 7.5))
    figure.tight_layout()
    plt.savefig('confusionmatrix_typeinfer.png')

    cm_gt_bt = confusion_matrix(gt_type, bt_type,
                                labels=subtype_names,
                                normalize='true')
    disp2 = ConfusionMatrixDisplay(cm_gt_bt,
                                   display_labels=subtype_names)
    disp2.plot(xticks_rotation='vertical')
    disp2.ax_.set_title('BERType confusion matrix.')
    figure = plt.gcf()
    figure.set_size_inches((7.5, 7.5))
    figure.tight_layout()
    plt.savefig('confusionmatrix_bertype.png')
