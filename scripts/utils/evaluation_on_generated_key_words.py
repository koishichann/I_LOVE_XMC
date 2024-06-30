from utils import read_index, read_label_text, load_map, transfer_indexs_to_labels
import os

dataset = 'wiki10-31k'
running_args = ['bart=5-stem-0.5-0.5',
                'two-bart=5-stem-0.5-0.5',
                #'two-bart-kpdrop_a=5-0.7-stem-0.5-0.5',
                #'two-bart-kpdrop_r=5-0.7-stem-0.5-0.5',
                #'two-bart-kpdrop_na=5-0.7-stem-0.5-0.5',
                'two-bart-kpdrop_nr=5-0.7-stem-0.5-0.5',
                #'two-bart-kpinsert_a=3-0.3-stem-0.5-0.5',
                #'two-bart-kpinsert_r=3-0.3-stem-0.5-0.5',
                #'two-bart-kpinsert_na=3-0.3-stem-0.5-0.5',
                'two-bart-kpinsert_nr=3-0.3-stem-0.5-0.5',
                'two-bart-kpdrop_nr-kpinsert_nr=3-0.7-0.3-stem-0.5-0.5'
                ]

for running_arg in running_args:
    pred_dir = os.path.join('..', '..', 'dataset', dataset, 'records', 'keyphrase_generation', 'res',
                            running_arg, 'tst_pred.txt')
    datadir = os.path.join('..', '..', 'dataset', dataset)
    indexes = read_index(os.path.join(datadir, "Y.tst.txt"))
    label_map = load_map(os.path.join(datadir, "output-items.txt"))
    labels_list = transfer_indexs_to_labels(label_map, indexes)  # list,需要转化成text
    if 'wiki' in datadir:
        for id1, label_list in enumerate(labels_list):
            for id2, label in enumerate(label_list):
                labels_list[id1][id2] = label.replace('_', ' ')
    pred_list = read_label_text(pred_dir)
    s = set()

    total_num = 0
    recall_num = 0
    for id, labels in enumerate(labels_list):
        s.clear()
        for label in labels:
            s.add(label)
        total_num = total_num + len(s)
        for pred_label in pred_list[id]:
            if pred_label in s:
                recall_num = recall_num + 1
    r = recall_num / total_num

    total_num = 0
    pred_num = 0
    for id, pred_labels in enumerate(pred_list):
        s.clear()
        for pred_label in pred_labels:
            s.add(pred_label)
        total_num = total_num + len(s)
        for label in labels_list[id]:
            if label in s:
                pred_num = pred_num + 1
    p = pred_num / total_num

    print('\n', running_arg, ':')
    print('precision:', p, '\trecall:', r, '\n')
