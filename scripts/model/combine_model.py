'''
This model is used to combine or matching . It provides least 4 types: cross-encoder,
'''
import os
import pickle
import torch
import math
from torch.utils.data import DataLoader
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from tqdm import tqdm
from ..utils.utils import load_map, read_label_text
from ..utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combine_train(model_name, model_save_dir, data_dir, num_epoch, batch_size):
    label_list = load_map(os.path.join(data_dir, "output-items.txt"))
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    train_data_un = [InputExample(texts=[i, i]) for m in label_list for i in m]
    # train_dataloader_un = DataLoader(fine_tune_unsup, shuffle=True, batch_size=batch_size)
    train_dataloader = DataLoader(train_data_un, batch_size=batch_size, shuffle=True)
    warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1)  # 10% of train data for warm-up
    #   logger.info("Warmup-steps: {}".format(warmup_steps))
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epoch,
              warmup_steps=warmup_steps,
              # train_loss=train_loss,
              # 用curr
              output_path=model_save_dir)
    model.save(model_save_dir)


###
def combine(args, type):
    ###
    pred_dir = os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME,
                            type + '_pred.txt')
    reference_dir = os.path.join(utils.BASE_DATASET_DIR, 'output-items.txt')
    model_name = args.combine_model_name
    data_dir = utils.COMBINE_RECORDS_DIR
    output_dir = os.path.join(utils.COMBINE_RECORDS_DIR, 'res', utils.COMBINE_MODEL_NAME,
                              type + '_combine.txt')
    if os.path.exists((output_dir)):
        return
    ###
    if not os.path.exists(pred_dir):
        print(f'{pred_dir} not exists, return. ')
        return
    model_b = SentenceTransformer(model_name_or_path=model_name, device=device)
    # model_b = SentenceTransformer('all-MiniLM-L12-v2',device=device)
    print(f'combine model_name: {model_name}')
    print('pred_data: ' + pred_dir)
    print('reference:' + reference_dir)
    print('write into: ' + output_dir)
    pred_list = read_label_text(pred_dir)
    all_label_list = load_map(reference_dir)
    if not os.path.exists(os.path.join(data_dir, 'all_labels.pkl')):
        embeddings_all = model_b.encode(all_label_list, convert_to_tensor=True, device=device)
        with open(os.path.join(data_dir, 'all_labels.pkl'), "wb") as fOut:
            pickle.dump({'embeddings': embeddings_all}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(data_dir, 'all_labels.pkl'), "rb") as fIn:
            stored_data = pickle.load(fIn)
            # stored_sentences = stored_data['sentences']
            embeddings_all = stored_data['embeddings']
    for i in tqdm(range(len(pred_list))):
        no_equal_list = []
        for ind, each_label in enumerate(pred_list[i]):
            # 此处可以考虑不换位置，但是会更复杂，不确定是否会影响结果，单纯extend的话有可能劣化结果
            if each_label not in all_label_list:
                no_equal_list.append({'ind': ind, 'label': each_label})
            # 对于每一个不在已存在标签列表中的label，计算得到最相似的标签
        if len(no_equal_list) == 0:
            continue
        t_list = list(map(lambda x: x['label'], no_equal_list))
        embeddings_pre = model_b.encode(t_list, convert_to_tensor=True, device=device)
        cosine_score = util.cos_sim(embeddings_pre, embeddings_all)
        # cosine_score是一个len(no_equal_list)行，(all_label_list)列的一个矩阵
        # cosine_score的长度一定等于no_equal_list
        flag = torch.zeros(len(all_label_list), device=device)
        for j in range(len(cosine_score)):
            this_score = torch.add(cosine_score[j], flag)
            max_ind = torch.argmax(this_score)
            # while all_label_list[max_ind] in pred_list: #if prelist has this candidate label
            #    cosine_score[j][max_ind]=0
            #    max_ind = cosine_score[j].argmax(0)
            no_equal_list[j]['label'] = all_label_list[max_ind]
            flag[max_ind] = -2.0
        for j in no_equal_list:
            pred_list[i][j['ind']] = j['label']
    # replace duplicate
    pre_new_list = []
    for i in range(len(pred_list)):
        tmp = []
        for j in range(len(pred_list[i])):
            if pred_list[i][j] not in tmp:
                tmp.append(pred_list[i][j])
        pre_new_list.append(tmp)
    if output_dir:
        print('write into: ' + output_dir)
        with open(output_dir, 'w+', encoding='utf-8', errors='ignore') as w1:
            for row in pre_new_list:
                w1.write(" || ".join(row) + '\n')
