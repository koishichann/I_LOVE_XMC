import os
from argparse import ArgumentParser, Namespace
import datetime
import sys
from functools import wraps
import logging
import time
import pickle
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import random
import nltk
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
########################################################################################################################
# define program constants
RUNNING_RECORDS_DIR = ""
BASE_DATASET_DIR = ""
BASE_RECORD_DIR = ""
KEYPHRASE_GENERATION_RECORDS_DIR = ""
KEYPHRASE_GENERATION_MODEL_NAME = ""
COMBINE_MODEL_NAME = ""
RANK_MODEL_NAME = ""
COMBINE_RECORDS_DIR = ""
RANK_RECORDS_DIR = ""
OUTPUT_TST_TEXT_DIR = ""
OUTPUT_TST_INDEX_DIR = ""
RES_OUTPUT_DIR = ""


def update_constants(args):
    global RUNNING_RECORDS_DIR
    RUNNING_RECORDS_DIR = os.path.join(args.log_pos)
    global BASE_DATASET_DIR
    BASE_DATASET_DIR = os.path.join(args.datadir, args.dataset)
    global BASE_RECORD_DIR
    BASE_RECORD_DIR = os.path.join(BASE_DATASET_DIR, 'records')
    global KEYPHRASE_GENERATION_RECORDS_DIR
    KEYPHRASE_GENERATION_RECORDS_DIR = os.path.join(BASE_RECORD_DIR, 'keyphrase_generation')
    global COMBINE_RECORDS_DIR
    COMBINE_RECORDS_DIR = os.path.join(BASE_RECORD_DIR, 'combine')
    global RANK_RECORDS_DIR
    RANK_RECORDS_DIR = os.path.join(BASE_RECORD_DIR, 'rank')
    global KEYPHRASE_GENERATION_MODEL_NAME
    KEYPHRASE_GENERATION_MODEL_NAME = args.kg_type + '=' + str(args.kg_epoch)
    if 'prefix' in args.kg_type:
        KEYPHRASE_GENERATION_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '-' + str(args.prefix_token_num)
    if 'kpdrop' in args.kg_type:
        KEYPHRASE_GENERATION_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '-' + str(args.kpdrop_rate)
    if 'kpinsert' in args.kg_type:
        KEYPHRASE_GENERATION_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '-' + str(args.kpinsert_rate)
    KEYPHRASE_GENERATION_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '-' + str(args.match)
    if 'stem' in args.match:
        KEYPHRASE_GENERATION_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '-' + str(args.stem_lambda) + '-' + str(
            args.stem_delta)
    global COMBINE_MODEL_NAME
    COMBINE_MODEL_NAME = KEYPHRASE_GENERATION_MODEL_NAME + '_' + args.combine_type
    global RANK_MODEL_NAME
    RANK_MODEL_NAME = COMBINE_MODEL_NAME + '_' + args.rank_type + '=' + str(args.rank_epoch)
    global OUTPUT_TST_TEXT_DIR
    OUTPUT_TST_TEXT_DIR = os.path.join(RANK_RECORDS_DIR, 'res', RANK_MODEL_NAME,
                                       'tst_rank_text.txt')
    global OUTPUT_TST_INDEX_DIR
    OUTPUT_TST_INDEX_DIR = os.path.join(RANK_RECORDS_DIR, 'res', RANK_MODEL_NAME,
                                        'tst_rank_index.txt')
    global RES_OUTPUT_DIR
    RES_OUTPUT_DIR = os.path.join(RANK_RECORDS_DIR, 'res', RANK_MODEL_NAME,
                                  'res.txt')


########################################################################################################################
# class to load args from terminal
def load_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./dataset',
                        help='dataset_dir')  # eurlex-4k, amazoncat-13k,wiki10-31k,wiki500k,amazon-3m,amazon-670k
    parser.add_argument('--dataset', type=str, default='eurlex-4k',
                        help='dataset_name')  # eurlex-4k, amazoncat-13k,wiki10-31k,wiki500k,amazon-3m,amazon-670k
    parser.add_argument('--log_pos', type=str, default='./logs', help='log_store_position')
    parser.add_argument('--kg_model_name', type=str, default='facebook/bart-base',
                        help='kg_model_name')
    parser.add_argument('--kg_type', type=str, default='bart')  # bart,pega,t5
    parser.add_argument('--combine_type', type=str, default='bi')  # bi, cr, del, sim
    parser.add_argument('--rank_type', type=str, default='bi')  # bi,cr,sim
    parser.add_argument('--kg_sw', type=str, default='pl')  # pl, hg
    parser.add_argument('--max_len', type=int, default=1024)  # ''tokenizer max length of document.
    parser.add_argument('--prefix_token_num', type=int, default=10)
    parser.add_argument('--kpdrop_rate', type=float, default=0.7)
    parser.add_argument('--kpinsert_rate', type=float, default=0.7)
    parser.add_argument('--match', type=str, default='exact', help='exact or stem match')
    parser.add_argument('--stem_lambda', type=float, default=0.5,
                        help='stem match: proportion of occurring words > lambda')
    parser.add_argument('--stem_delta', type=float, default=0.5, help='stem match: cosine similarity > delta')
    parser.add_argument('--stem_model', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='the model used for stem match checking cosine similarity')

    # finetune args
    parser.add_argument('--is_kg_train', type=int, default=1,
                        help="whether run finteune processing")

    parser.add_argument('-b', '--kg_batch_size', type=int, default=4,
                        help='number of batch size for training')
    parser.add_argument('-e', '--kg_epoch', type=int, default=3,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--kg_checkdir', type=str, default='bart_check',
                        help='path to trained model to save')
    parser.add_argument('--kg_savedir', type=str, default='bart_save',
                        help="fine-tune model save dir")
    parser.add_argument('--kg_lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--kg_seed', type=int, default=44,
                        help='random seed (default: 1)')
    # parser.add_argument('--kg_trn_data', type=str, default='Y.trn.txt')
    # parser.add_argument('--kg_tst_data', type=str, default='Y.tst.txt')

    # perdicting args
    parser.add_argument('--is_kg_pred', type=int, default=1, help="whether predict")
    parser.add_argument('--is_pred_trn', type=int, default=1,
                        help="Whether run predicting training dataset")
    parser.add_argument('--is_pred_tst', type=int, default=1,
                        help="Whether run predicting testing dataset")
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.75)
    # combine part
    parser.add_argument('--is_combine', type=int, default=1,
                        help="Whether run combine")
    parser.add_argument('--combine_model_name', type=str, default='sentence-transformers/all-MiniLM-L12-v2')

    # rank part
    parser.add_argument('--is_rank_train', type=int, default=1)
    parser.add_argument('--rank_model', type=str, default='sentence-transformers/all-MiniLM-L12-v2')
    parser.add_argument('--rank_batch', type=int, default=64)
    parser.add_argument('--rank_epoch', type=int, default=3)
    parser.add_argument('--rank_model_save', type=str, default='bi_rank_save')
    parser.add_argument('--is_rank', type=int, default=1)
    parser.add_argument('--is_p_at_k', type=int, default=1)

    args = parser.parse_args()
    update_constants(args)

    ########################################################################################################################
    # define logfile and output position
    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    t = datetime.datetime.now()
    logfilename = os.path.join(RUNNING_RECORDS_DIR,
                               "log_" + str(t.year) + '-' + str(t.month) + '-' + str(t.day) + '_' + str(
                                   t.hour) + '_' + str(
                                   t.minute) + '.txt')
    print('logfile save in:', logfilename)
    sys.stdout = Logger(logfilename, sys.stdout)
    sys.stderr = Logger(logfilename, sys.stderr)  # redirect std err, if necessary
    return args


########################################################################################################################
# function to log information
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__file__)


def func_log(func):
    t_start = time.time()

    @wraps(func)
    def decorated(*args, **kwargs):
        print("call %s():" % func.__name__)
        for i in args:
            # print(i)
            logger.info(i)
        return func(*args, **kwargs)

    t_end = time.time()
    logger.info(f'time cost {t_end - t_start} sec.')
    return decorated


def record_finish_time(output_dir):
    s = str(datetime.datetime.now())
    print('recording finish time: ' + s)
    with open(output_dir, 'w+', encoding='utf-8', errors='ignore') as t:
        t.write('finish running this part at ' + s)


########################################################################################################################
# functions to process data

def judge_present_label(text, label, match, stem_lambda, stem_delta, text_stems, text_embedding, model_b,
                        label_word_num_map):
    label_stems = label.split(" ")
    word_num = len(label_stems)
    if label in text:
        return True, word_num
    if match == 'exact':
        return False, word_num
    label_stems = [ps.stem(w) for w in label_stems]
    if word_num == 1:
        return label_stems[0] in text_stems, word_num
    occurring_num = 0.0
    for stem in label_stems:
        if stem in text_stems:
            occurring_num = occurring_num + 1.0
    occurring_proportion = occurring_num / float(len(label_stems))
    if occurring_proportion < stem_lambda:
        return False, word_num
    label_embedding = model_b.encode(label, convert_to_tensor=True, device=device)
    cosine_score = util.cos_sim(label_embedding, text_embedding)
    return cosine_score > stem_delta, word_num


def separate_present_absent_labels(indexes, labels_list, texts, match, stem_lambda, stem_delta, stem_model):
    present_labels, absent_labels = [], []
    present_indexes, absent_indexes = [], []
    present_num, absent_num, all_num = 0.0, 0.0, 0.0
    model_b = None
    label_word_num_map = {}
    if match == 'stem':
        model_b = SentenceTransformer(model_name_or_path=stem_model, device=device)
    for i, index in enumerate(indexes):
        present_label, absent_label = [], []
        present_index, absent_index = [], []
        text_stems = []
        text_embedding = None
        if match == 'stem':
            text = ' '.join(texts[i])
            src_sentences = nltk.sent_tokenize(text)
            src_sentences = [src_sentence.split(" ") for src_sentence in src_sentences]
            for src_sentence in src_sentences:
                for w in src_sentence:
                    text_stems.append(ps.stem(w))
            text_embedding = model_b.encode(texts[i], convert_to_tensor=True, device=device)
        for j, label in enumerate(labels_list[i]):
            judge_result, word_num = judge_present_label(texts[i], label, match, stem_lambda, stem_delta, text_stems,
                                                         text_embedding,
                                                         model_b, label_word_num_map)
            if label_word_num_map.get(word_num) is None:
                label_word_num_map[word_num] = {'all_num': 0.0, 'present_num': 0.0, 'present_ratio': 0.0,
                                                'absent_num': 0.0, 'absent_ratio': 0.0}
            if judge_result:
                present_label.append(label)
                present_index.append(index[j])
                present_num = present_num + 1.0
                label_word_num_map[word_num]['present_num'] = label_word_num_map[word_num]['present_num'] + 1.0
            else:
                absent_label.append(label)
                absent_index.append(index[j])
                absent_num = absent_num + 1.0
                label_word_num_map[word_num]['absent_num'] = label_word_num_map[word_num]['absent_num'] + 1.0
            all_num = all_num + 1.0
            label_word_num_map[word_num]['all_num'] = label_word_num_map[word_num]['all_num'] + 1.0
        present_labels.append(present_label)
        absent_labels.append(absent_label)
        present_indexes.append(present_index)
        absent_indexes.append(absent_index)
    present_ratio, absent_ratio = present_num / all_num, absent_num / all_num
    for i in label_word_num_map:
        label_word_num_map[i]['present_ratio'] = label_word_num_map[i]['present_num'] / label_word_num_map[i]['all_num']
        label_word_num_map[i]['absent_ratio'] = label_word_num_map[i]['absent_num'] / label_word_num_map[i]['all_num']
    label_word_num_map[20020111] = {'all_num': all_num, 'present_num': present_num, 'present_ratio': present_ratio,
                                    'absent_num': absent_num, 'absent_ratio': absent_ratio}
    return present_labels, absent_labels, present_indexes, absent_indexes, present_ratio, absent_ratio, label_word_num_map


def kpdrop(present_labels: list, absent_labels: list, present_texts: list, absent_texts: list, texts: list, kpdrop_type,
           kpdrop_rate):
    l = len(texts)
    new_present_texts, new_absent_texts = present_texts.copy(), absent_texts.copy()
    new_present_labels, new_absent_labels = present_labels.copy(), absent_labels.copy()
    for i in range(l):
        new_present_label, new_absent_label = [], absent_labels[i]
        new_text = texts[i]
        for word in present_labels[i]:
            if random.random() < kpdrop_rate:
                new_text = new_text.replace(' ' + word + ' ', ' <mask> ')
                new_text = new_text.replace(' ' + word + '.', ' <mask>.')
                new_text = new_text.replace(' ' + word + ',', ' <mask>,')
                new_text = new_text.replace(' ' + word + '!', ' <mask>!')
                new_text = new_text.replace(' ' + word + '?', ' <mask>?')
                new_absent_label.append(word)
            else:
                new_present_label.append(word)
        if kpdrop_type == 'a':
            new_present_texts.append(new_text)
            new_absent_texts.append(new_text)
            new_present_labels.append(new_present_label)
            new_absent_labels.append(new_absent_label)
        elif kpdrop_type == 'r':
            new_present_texts[i] = new_text
            new_absent_texts[i] = new_text
            new_present_labels[i] = new_present_label
            new_absent_labels[i] = new_absent_label
        elif kpdrop_type == 'na':
            new_present_texts.append(texts[i])
            new_absent_texts.append(new_text)
            new_present_labels.append(present_labels[i])
            new_absent_labels.append(new_absent_label)
        elif kpdrop_type == 'nr':
            # new_present_texts[i] = texts[i]
            new_absent_texts[i] = new_text
            # new_present_labels[i] = present_labels[i]
            new_absent_labels[i] = new_absent_label

    return new_present_labels, new_absent_labels, new_present_texts, new_absent_texts


# 可以尝试一种新的kpdrop：用所有的present训练present，用所有的present+absent训练absent


def kpinsert(present_labels: list, absent_labels: list, present_texts: list, absent_texts: list, texts: list,
             kpinsert_type, kpinsert_rate, max_len):
    l = len(texts)
    new_present_texts, new_absent_texts = present_texts.copy(), absent_texts.copy()
    new_present_labels, new_absent_labels = present_labels.copy(), absent_labels.copy()
    overflow = 0.0
    no_overflow = 0.0
    for i in range(l):
        new_present_label, new_absent_label = present_labels[i], []
        new_text: str = "Topic: "
        for word in absent_labels[i]:
            if random.random() < kpinsert_rate:
                new_text = new_text + word + ", "
                new_present_label.append(word)
            else:
                new_absent_label.append(word)
        new_text = new_text.rstrip(', ')
        if len(new_text) > max_len:
            overflow = overflow + 1.0
        else:
            no_overflow = no_overflow + 1.0
        new_text = new_text + '. ' + texts[i]
        if kpinsert_type == 'a':
            new_present_texts.append(new_text)
            new_absent_texts.append(new_text)
            new_present_labels.append(new_present_label)
            new_absent_labels.append(new_absent_label)
        elif kpinsert_type == 'r':
            new_present_texts[i] = new_text
            new_absent_texts[i] = new_text
            new_present_labels[i] = new_present_label
            new_absent_labels[i] = new_absent_label
        elif kpinsert_type == 'na':
            new_present_texts.append(texts[i])
            new_absent_texts.append(new_text)
            new_present_labels.append(new_present_label)
            new_absent_labels.append(absent_labels[i])
        elif kpinsert_type == 'nr':
            new_present_texts[i] = new_text
            # new_absent_texts[i] = text
            new_present_labels[i] = new_present_label
            # new_absent_labels[i] = absent_labels[i]

    return new_present_labels, new_absent_labels, new_present_texts, new_absent_texts, overflow / (
                overflow + no_overflow)


# 如需要kpinsert和kpdrop同时使用，则应该都用nr选项

def add_shuffle_examples(labels_list, texts):
    l = len(labels_list)
    for i in range(l):
        new_labels = []
        if len(labels_list[i]) >= 2:
            texts.append(texts[i])
            new_labels = labels_list[i].copy()
            new_labels.reverse()
            labels_list.append(new_labels)
        if len(labels_list[i]) >= 3:
            texts.append(texts[i])
            new_labels = labels_list[i].copy()
            temp = new_labels[0]
            new_labels[0] = new_labels[-1]
            new_labels[-1] = temp
            labels_list.append(new_labels)


@func_log
def read_text(src) -> List:
    res = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as r:
        for i in r:
            res.append(i.strip())
    return res


@func_log
def read_index(src) -> List[List[int]]:
    res = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as r:
        for i in r:
            cur = []
            for j in i.strip().split(","):
                cur.append(int(j))
            res.append(cur)
    return res


@func_log
def read_label_text(src) -> List[List[str]]:
    res = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as r:
        for i in r:
            if 'k%c3%a4sikirjoittaminen' in i:
                pass
            res.append(i.strip().split(" || "))
    return res


@func_log
def load_map(src) -> List[str]:
    '''
    load label index map
    src = ./dataset/data-name/output-items.txt
    return  label List, which has label index information.
    '''
    label_map = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as r:
        for i in r:
            label_map.append(i.strip())
    return label_map


# @func_log
def transfer_indexs_to_labels(label_map, index_lists) -> List[List[str]]:
    label_texts = []
    for i in index_lists:
        cur_labels = []  # 对于一条记录的labels 做映射
        for j in i:
            cur_labels.append(label_map[j])
        label_texts.append(cur_labels)
    return label_texts


def transfer_labels_to_index(label_map: List[str], label_texts) -> List[List[str]]:
    index_list = []
    for i in label_texts:
        cur_indexs = []
        for j in i:
            cur_indexs.append(label_map.index(j))
        index_list.append(cur_indexs)
    return index_list


@func_log
def p_at_k(src_label_dir, pred_label_dir, outputdir=None) -> list:
    # src_label_dir = dir+src_label_dir
    # pred_label_dir = os.path.join(dir,'res',pred_label_dir)
    print("p_at_k:" + '\n')
    print("src_label: " + src_label_dir)
    print("pred_label: " + pred_label_dir)
    p_at_1_count = 0
    p_at_3_count = 0
    p_at_5_count = 0
    src_label_list = []
    pred_label_list = []
    src_label_list = read_index(src_label_dir)
    pred_label_list = read_index(pred_label_dir)
    num1 = len(src_label_list)
    num2 = len(pred_label_list)
    if num1 != num2:
        print("num error")
        return
    else:
        # recall_100 = get_recall_100(src_label_list,pred_label_list)
        for i in range(num1):
            p1 = 0
            p3 = 0
            p5 = 0
            for j in range(len(pred_label_list[i])):
                if pred_label_list[i][j] in src_label_list[i]:
                    if j < 1:
                        p1 += 1
                        p3 += 1
                        p5 += 1
                    if j >= 1 and j < 3:
                        p3 += 1
                        p5 += 1
                    if j >= 3 and j < 5:
                        p5 += 1
            p_at_1_count += p1
            p_at_3_count += p3
            p_at_5_count += p5
        p1 = p_at_1_count / len(pred_label_list)
        p3 = p_at_3_count / (3 * len(pred_label_list))
        p5 = p_at_5_count / (5 * len(pred_label_list))
        print('p@1= ' + str(p1))
        print('p@3= ' + str(p3))
        print('p@5= ' + str(p5))
        # print(f'recall@100 = {recall_100:>4f}')
        if outputdir:
            with open(outputdir, 'a+') as w:
                w.write("\n")
                # now_time = datetime.datetime.now()
                # time_str = now_time.strftime('%Y-%m-%d %H:%M:%S')
                # w.write("time: "+time_str+"\n")
                w.write("src_label: " + src_label_dir + "\n")
                w.write('pred_label: ' + pred_label_dir + "\n")
                w.write("p@1=" + str(p1) + "\n")
                w.write("p@3=" + str(p3) + "\n")
                w.write("p@5=" + str(p5) + "\n")
                # w.write(f"recall@100={recall_100:>4f}")
        return [p1, p3, p5]


@func_log
def p_at_k_text(dir, src_label_dir, pred_label_dir, outputdir=None) -> list:
    # src_label_dir = dir+src_label_dir
    # pred_label_dir = os.path.join(dir,'res',pred_label_dir)
    print("p_at_k:" + '\n')
    print("src_label: " + src_label_dir)
    print("pred_label: " + pred_label_dir)
    p_at_1_count = 0
    p_at_3_count = 0
    p_at_5_count = 0
    src_label_list = []
    pred_label_list = []
    src_label_list = read_label_text(src_label_dir)
    pred_label_list = read_label_text(pred_label_dir)
    num1 = len(src_label_list)
    num2 = len(pred_label_list)
    if num1 != num2:
        print("num error")
        return
    else:
        # recall_100 = get_recall_100(src_label_list,pred_label_list)
        for i in range(num1):
            p1 = 0
            p3 = 0
            p5 = 0
            for j in range(len(pred_label_list[i])):
                if pred_label_list[i][j] in src_label_list[i]:
                    if j < 1:
                        p1 += 1
                        p3 += 1
                        p5 += 1
                    if j >= 1 and j < 3:
                        p3 += 1
                        p5 += 1
                    if j >= 3 and j < 5:
                        p5 += 1
            p_at_1_count += p1
            p_at_3_count += p3
            p_at_5_count += p5
        p1 = p_at_1_count / len(pred_label_list)
        p3 = p_at_3_count / (3 * len(pred_label_list))
        p5 = p_at_5_count / (5 * len(pred_label_list))
        print('p@1= ' + str(p1))
        print('p@3= ' + str(p3))
        print('p@5= ' + str(p5))
        # print(f'recall@100 = {recall_100:>4f}')
        if outputdir:
            with open(outputdir, 'a+') as w:
                w.write("\n")
                # now_time = datetime.datetime.now()
                # time_str = now_time.strftime('%Y-%m-%d %H:%M:%S')
                # w.write("time: "+time_str+"\n")
                w.write("src_label: " + src_label_dir + "\n")
                w.write('pred_label: ' + pred_label_dir + "\n")
                w.write("p@1=" + str(p1) + "\n")
                w.write("p@3=" + str(p3) + "\n")
                w.write("p@5=" + str(p5) + "\n")
                # w.write(f"recall@100={recall_100:>4f}")
        return [p1, p3, p5]


def construct_rank_train(data_dir, model_name, label_map_dir, ground_index_dir, src_text_dir, output_index=None,
                         output_label=None):
    '''
    调用untrained simces或者sentence-transformer排序所有的labels，然后选取前10/5个语义高度相关但是negetive的label作为负标签
    '''
    print(f'label_map: {label_map_dir}')
    print(f'ground_index_dir: {ground_index_dir}')
    print(f'src_text_dir: {src_text_dir}')
    ground_index = read_index(ground_index_dir)
    src_text = read_text(src_text_dir)
    with open(data_dir + 'all_labels.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        # stored_sentences = stored_data['sentences']
        embeddings_all = stored_data['embeddings']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    embeddings_src = model.encode(src_text, convert_to_tensor=True, device=device)
    cos_scores = util.cos_sim(embeddings_src, embeddings_all)
    un_contain_list = []
    res = []
    for i in tqdm(range(len(cos_scores))):
        count = 7
        '''调整为5可以减少negative num数量'''
        tmp = []
        flag = torch.zeros(len(embeddings_all), device=device)
        while count > 0:
            this_score = torch.add(cos_scores[i], flag)
            max_ind = torch.argmin(this_score).item()
            if max_ind not in ground_index[i]:
                '''不在，说明是negative'''
                tmp.append(max_ind)
                count -= 1
            '''是ground，直接标记以下就可以了'''
            flag[max_ind] = 2.0
        un_contain_list.append(tmp)
    for i in range(len(ground_index)):
        ground_index[i].extend(un_contain_list[i])
    if output_index:
        with open(output_index, 'w') as w:
            for row in ground_index:
                w.write(" || ".join(map(lambda x: str(x), row)) + "\n")
    return res
