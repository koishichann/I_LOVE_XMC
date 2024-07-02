# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import datetime
import os
import sys
import torch
import wandb
import platform
from scripts.utils import utils
from scripts.model.keyphrase_generation_model import kg_train, kg_predict, KG_Model, data_preprocess
from scripts.model.combine_model import combine
from scripts.model.rank_model import rank_train, label_rank

device = 'cuda' if torch.cuda.is_available() else 'cpu'


########################################################################################################################
# main process
def run(args):
    start_time = datetime.datetime.now()
    print('args:', args)
    wandb.init(project="I_Love_XMC", name=platform.node() + ':' + args.dataset + ' ' + str(datetime.datetime.now()),
               notes=str(args))
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')

    if not os.path.exists(path=os.path.join(utils.BASE_RECORD_DIR)):
        os.mkdir(path=os.path.join(utils.BASE_RECORD_DIR))

    print('part1: keyphrase generation\n\n')
    if not os.path.exists(path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR)):
        os.mkdir(path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR))
    if not os.path.exists(path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res')):
        os.mkdir(path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res'))
    if not os.path.exists(
            path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME)):
        os.mkdir(
            path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME))
    ###
    # if not os.path.exists(path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'finish_flag.txt')):
    data_preprocess(args)
    train_present_ratio, train_absent_ratio, val_present_ratio, val_absent_ratio = 0.0, 0.0, 0.0, 0.0
    if args.is_kg_train:
        # train_present_ratio, train_absent_ratio, val_present_ratio, val_absent_ratio = (
        kg_train(args=args)  # )
        # model = (KG_Model.load_from_checkpoint(checkpoint_path=os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR,
        #                                                                    utils.KEYPHRASE_GENERATION_MODEL_NAME)).to(
        #    device))
        model = KG_Model(args=args)
        if model.two:
            model.present_model.load_state_dict(torch.load(os.path.join(
                utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME,
                'model_save_present')), strict=False)
            model.absent_model.load_state_dict(torch.load(os.path.join(
                utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME,
                'model_save_absent')), strict=False)
        else:
            model.model.load_state_dict(torch.load(os.path.join(
                utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'res', utils.KEYPHRASE_GENERATION_MODEL_NAME, 'model_save')),
                strict=False)
    else:
        model = KG_Model(args=args)
    if args.is_kg_pred:
        if args.is_pred_trn:
            kg_predict(model=model, args=args, type='trn')
        if args.is_pred_tst:
            kg_predict(model=model, args=args, type='tst')
    # utils.record_finish_time(os.path.join(utils.KEYPHRASE_GENERATION_RECORDS_DIR, 'finish_flag.txt'))
    ###
    print('keyphrase generation finished\n\n')

    print('part 2: check combine args, combine\n\n')
    if not os.path.exists(path=os.path.join(utils.COMBINE_RECORDS_DIR)):
        os.mkdir(path=os.path.join(utils.COMBINE_RECORDS_DIR))
    if not os.path.exists(path=os.path.join(utils.COMBINE_RECORDS_DIR, 'res')):
        os.mkdir(path=os.path.join(utils.COMBINE_RECORDS_DIR, 'res'))
    if not os.path.exists(path=os.path.join(utils.COMBINE_RECORDS_DIR, 'res', utils.COMBINE_MODEL_NAME)):
        os.mkdir(path=os.path.join(utils.COMBINE_RECORDS_DIR, 'res', utils.COMBINE_MODEL_NAME))
    # if not os.path.exists(path=os.path.join(utils.COMBINE_RECORDS_DIR, 'finish_flag.txt')):
    if args.is_combine:
        if args.is_pred_trn:
            combine(args=args, type='trn')
        if args.is_pred_tst:
            combine(args=args, type='tst')
        # utils.record_finish_time(os.path.join(utils.COMBINE_RECORDS_DIR, 'finish_flag.txt'))
    print('combine finished\n\n')

    print('part 3: check rank args, rank train, rank\n\n')
    if not os.path.exists(path=os.path.join(utils.RANK_RECORDS_DIR)):
        os.mkdir(path=os.path.join(utils.RANK_RECORDS_DIR))
    if not os.path.exists(path=os.path.join(utils.RANK_RECORDS_DIR, 'res')):
        os.mkdir(path=os.path.join(utils.RANK_RECORDS_DIR, 'res'))
    if not os.path.exists(path=os.path.join(utils.RANK_RECORDS_DIR, 'res', utils.RANK_MODEL_NAME)):
        os.mkdir(path=os.path.join(utils.RANK_RECORDS_DIR, 'res', utils.RANK_MODEL_NAME))
    if args.is_rank_train:
        rank_train(args=args, type='trn')
    if args.is_rank:
        label_rank(args=args, type='tst')
    if args.is_p_at_k:
        res_list = utils.p_at_k(src_label_dir=os.path.join(utils.BASE_DATASET_DIR, 'Y.tst.txt'),
                                pred_label_dir=utils.OUTPUT_TST_INDEX_DIR, outputdir=utils.RES_OUTPUT_DIR)
        with open(os.path.join(utils.BASE_DATASET_DIR, 'result_logs.txt'), 'a') as r:
            end_time = datetime.datetime.now()
            r.write('\nstart time:' + str(start_time) + '\n')
            r.write('end time:' + str(end_time) + '\n')
            r.write('program cost:' + str(end_time - start_time) + '\n')
            r.write(str(args) + '\n')
            r.write('P@1:' + str(res_list[0]) + '\n')
            r.write('P@3:' + str(res_list[1]) + '\n')
            r.write('P@5:' + str(res_list[2]) + '\n')
            # r.write('train_present_ratio:' + str(train_present_ratio) + '\ttrain_absent_ratio' + str(
            #    train_absent_ratio) + '\tval_present_ratio' + str(val_present_ratio) + '\tval_absent_ratio' + str(
            #    val_absent_ratio) + '\n')
    print('rank finished\n\n')


if __name__ == '__main__':
    args = utils.load_args()

    print('\n\n### program starting ###\n\n')
    print('RUNNING_RECORDS_DIR:' + utils.RUNNING_RECORDS_DIR)
    print('BASE_DATASET_DIR:' + utils.BASE_DATASET_DIR)
    print('BASE_RECORD_DIR:' + utils.BASE_RECORD_DIR)
    print('KEYPHRASE_GENERATION_RECORDS_DIR:' + utils.KEYPHRASE_GENERATION_RECORDS_DIR)
    print('COMBINE_RECORDS_DIR:' + utils.COMBINE_RECORDS_DIR)
    run(args)

    print('\n\n### program ending ###\n\n')
    wandb.alert(text='succeeding finish running', title='running ended')
