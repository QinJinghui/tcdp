# -*- encoding: utf-8 -*-
# @Author: Jinghui Qin
# @Time: 2021/12/22
# @File: run_pretrain_crd_and_bert2tree.py.py
import os
import glob
import shutil
import time
import math
import json
import pprint
import argparse
import torch.optim
from tqdm import tqdm
from src.MoCo import *
from src.models import *
from src.pre_data import *
from src.train_and_evaluate import *
from src.expressions_transfer import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

bert_path = "./pretrained_lm/chinese-roberta-wwm-ext"  # chinese-roberta-wwm-ext  # chinese-bert-wwm
# pretrained_path = "pretrained_model/"

def set_args():
    parser = argparse.ArgumentParser(description = "bert2tree")
    parser.add_argument('--pt_n_epochs', type=int, default=60)
    # parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--pt_seed', type=int, default=111)
    parser.add_argument('--pt_max_seq_length', type=int, default=180)
    parser.add_argument('--pt_batch_size', type=int, default=8)
    parser.add_argument('--pt_loss_weight', nargs='+', type=float, default=[0.0, 1.0, 1.0, 1.0])  # 0.15, 0.15, 1, 2000
    # parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--pt_hidden_size', type=int, default=768)
    parser.add_argument('--pt_learning_rate_bert', type=float, default=1e-4)
    parser.add_argument('--pt_weight_decay_bert', type=float, default=0.1)
    parser.add_argument('--pt_warmup_proportion', type=float, default=0.1)
    parser.add_argument('--pt_learning_rate', type=float, default=1e-4)
    parser.add_argument('--pt_weight_decay', type=float, default=1e-5)

    parser.add_argument('--pt_debug', action='store_true', default=False)

    tmp = parser.parse_args()
    parser.add_argument('--pt_save_path', type=str, default="model_CRD_4loss_lr%.0e_ep%d_wd%.0e_lrb%.0e_wdb%.0e_%f_%f_%f_%f"
                                                         %(tmp.pt_learning_rate, tmp.pt_n_epochs, tmp.pt_weight_decay, tmp.pt_learning_rate_bert, tmp.pt_weight_decay_bert,
                                                           tmp.pt_loss_weight[0],tmp.pt_loss_weight[1],tmp.pt_loss_weight[2],tmp.pt_loss_weight[3]))
    parser.add_argument('--pt_ori_path', type=str, default='./data/')
    parser.add_argument('--pt_prefix', type=str, default='23k_processed.json')
    parser.add_argument('--pt_raw_data_path', type=str, default="data/Math_23K.json")
    parser.add_argument('--pt_our_data_path', type=str, default="data/Math_23K_Noun_template_no_simplify.json")
    parser.add_argument('--pt_student_path', type=str, default="./pretrained_lm/chinese-roberta-wwm-ext" )
    parser.add_argument('--pt_teacher_path', type=str, default="teacher_model/")

    ### FineTune
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--step_size', type=int, default=25)
    # parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--max_seq_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--learning_rate_bert', type=float, default=5e-5)
    parser.add_argument('--weight_decay_bert', type=float, default=1e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    # parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--MaskN', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)

    tmp = parser.parse_args()
    parser.add_argument('--save_path', type=str, default="./model_math23k_traintest_lr%.0e_ep%d_wd%.0e_lrb%.0e_wdb%.0e_%f_%f_%f_%f"
                                                         %(tmp.learning_rate, tmp.n_epochs, tmp.weight_decay, tmp.learning_rate_bert, tmp.weight_decay_bert,
                                                           tmp.pt_loss_weight[0],tmp.pt_loss_weight[1],tmp.pt_loss_weight[2],tmp.pt_loss_weight[3]))
    parser.add_argument('--ori_path', type=str, default='./data/')
    parser.add_argument('--prefix', type=str, default='23k_processed.json')
    parser.add_argument('--raw_data_path', type=str, default="data/Math_23K.json")
    parser.add_argument('--our_data_path', type=str, default="data/Math_23K_Noun_template_no_simplify.json")
    parser.add_argument('--bert_path', type=str, default="./pretrained_model_math23k_traintest_lr%.0e_ep%d_wd%.0e_lrb%.0e_wdb%.0e_%f_%f_%f_%f"
                                                         %(tmp.learning_rate, tmp.n_epochs, tmp.weight_decay, tmp.learning_rate_bert, tmp.weight_decay_bert,
                                                           tmp.pt_loss_weight[0],tmp.pt_loss_weight[1],tmp.pt_loss_weight[2],tmp.pt_loss_weight[3]))

    args = parser.parse_args()
    return args


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = set_args()
    # 创建save文件夹
    if not os.path.exists(args.pt_save_path):
        os.makedirs(args.pt_save_path)
        print("make dir ", args.pt_save_path)


    # 设置随机种子
    setup_seed(args.pt_seed)

    train_fold, test_fold, valid_fold, generate_nums, copy_nums = process_data_pipeline(args.pt_raw_data_path,
                                                                                        args.pt_our_data_path,
                                                                                        args.pt_ori_path,
                                                                                        args.pt_prefix, debug=args.pt_debug)

    train_steps = args.pt_n_epochs * math.ceil(len(train_fold) / args.pt_batch_size)
    # print(train_steps)

    """
    data格式:{
        "tokens":tokens,
        "tokens_MaskN":tokens_MaskN,
        "num_idx":num_idx,
        "noun":[noun[key] for key in noun], 
        "name":[name[key] for key in name],
        "token_ids": ,
        "token_type_ids": ,
        "attention_mask": ,
        "output": ,
        "num_stack": ,
        "nums": ,
        "id": ,
        "original_text": ,
        "template": ,
    }
    """
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(train_fold, test_fold, 5, generate_nums,
                                                                    copy_nums, args.pt_student_path, args.pt_max_seq_length, tree=True)
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    # pp = pprint.PrettyPrinter(indent=4)
    # print(train_pairs[100])
    print([key for key in train_pairs[100]])

    # Initialize models
    student = Pretrain_Bert_CRD(bert_path=args.pt_student_path)
    student_k = Pretrain_Bert_CRD(bert_path=args.pt_student_path)
    teacher = Pretrain_Bert_CRD(bert_path=args.pt_teacher_path)
    student_cls = Bert_classification_head(hidden_size=student.hidden_size)

    param_optimizer = list(student.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.pt_weight_decay_bert},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_optimizer = AdamW(optimizer_grouped_parameters,
                              lr = args.pt_learning_rate_bert, # args.pt_learning_rate - default is 5e-5
                              eps = 1e-8, # args.pt_adam_epsilon  - default is 1e-8.
                              correct_bias = False
                              )
    student_scheduler = get_linear_schedule_with_warmup(student_optimizer,
                                                        num_warmup_steps = int(train_steps * args.pt_warmup_proportion), # Default value in run_glue.py
                                                        num_training_steps = train_steps)

    student_cls_optimizer = torch.optim.Adam(student_cls.parameters(), lr=args.pt_learning_rate, weight_decay=args.pt_weight_decay)
    student_cls_scheduler = torch.optim.lr_scheduler.StepLR(student_cls_optimizer, step_size=25, gamma=0.5)

    print(len(train_pairs))


    # # Move models to GPU
    if USE_CUDA:
        student.cuda()
        student_k.cuda()
        teacher.cuda()
        student_cls.cuda()

        # student_head.cuda()
        # teacher_head.cuda()

    train_data = prepare_data_Batch(train_pairs, args.pt_batch_size)
    myMoCo = MySupervisedMoCoCRD(student, student_k, teacher, ori_len=len(train_pairs), K=100 if args.pt_debug else len(train_pairs)) # 10000
    myMoCo.initialize_moco_queue(train_data)
    myMoCo.initialize_crd_queue(train_data)

    # 在使用完teacher后就应该抛弃, 减少显存消耗
    # del teacher
    # torch.cuda.empty_cache()
    min_loss = 9999

    for epoch in range(args.pt_n_epochs):
        random.seed(epoch + args.pt_seed)

        start = time.time()

        print(len(train_pairs))
        train_data = prepare_data_Batch(train_pairs, args.pt_batch_size)

        print("epoch:", epoch + 1)
        total_loss = 0
        total_cls_loss = 0
        total_con_loss = 0
        total_crd_loss = 0
        total_distill_loss = 0

        pbar = tqdm(train_data)
        for batch in pbar:
            loss = pretrain_my_CRD(myMoCo, student, teacher, student_k, student_cls, student_optimizer,
                                student_scheduler, student_cls_optimizer, batch, loss_weight=args.pt_loss_weight)

            total_loss += loss["loss"]
            total_cls_loss += loss["cls_loss"]
            total_con_loss += loss["con_loss"]
            total_crd_loss += loss["crd_loss"]
            total_distill_loss += loss["distill_loss"]

            pbar.set_description("## Loss: %f %f %f %f %f"%(loss["loss"], loss["cls_loss"], loss["con_loss"], loss["crd_loss"], loss["distill_loss"]))

        student_cls_scheduler.step()

        # myMoCo.momentum_update_queue()

        # 清空无用变量节约内存
        torch.cuda.empty_cache()

        total_loss /= len(train_data)
        total_cls_loss /= len(train_data)
        total_con_loss /= len(train_data)
        total_crd_loss /= len(train_data)
        total_distill_loss /= len(train_data)

        print("Loss: ", total_loss, total_cls_loss, total_con_loss, total_crd_loss, total_distill_loss)
        print("training time", time_since(time.time() - start))
        print("--------------------------------")

        if epoch%4 == 0 or epoch == args.pt_n_epochs - 1:
            student.savebert(args.pt_save_path + "/pytorch_model_epoch" + str(epoch) + "_" + str(total_loss) + ".bin")
            # if epoch == args.pt_n_epochs - 1:
            #     student.savebert(args.pt_save_path + "/pytorch_model.bin")
            #     final_path = args.pt_save_path + "/pytorch_model.bin"

        if total_loss < min_loss:
            print("**saving model in ", args.pt_save_path)
            print("——————————————————————————————————————")
            min_loss = total_loss
            student.savebert(args.pt_save_path + "/pytorch_model.bin")
            final_path = args.pt_save_path + "/pytorch_model.bin"

    ###############################################################
    # 初始化finetune环境
    if not os.path.exists(args.bert_path):
        os.makedirs(args.bert_path)
        print("make dir ", args.bert_path)

    file_pathes = glob.glob(bert_path+'/*')
    for file_path in file_pathes:
        if ".bin" not in file_path:
            shutil.copyfile(file_path, args.bert_path+'/'+file_path.split('/')[-1])

    shutil.copyfile(final_path, args.bert_path+'/'+final_path.split('/')[-1])

    ###################FineTune####################################
    # 创建save文件夹
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("make dir ", args.save_path)

    log_writer = SummaryWriter()
    # 设置随机种子
    setup_seed(args.seed)
    train_fold, test_fold, valid_fold, generate_nums, copy_nums = process_data_pipeline(args.raw_data_path, args.our_data_path, args.ori_path, args.prefix, debug=args.debug)

    train_steps = args.n_epochs * math.ceil(len(train_fold) / args.batch_size)
    # print(train_steps)

    """
    data格式:{
        "tokens":tokens,
        "tokens_MaskN":tokens_MaskN,
        "num_idx":num_idx,
        "noun":[noun[key] for key in noun], 
        "name":[name[key] for key in name],
        "token_ids": ,
        "token_type_ids": ,
        "attention_mask": ,
        "output": ,
        "num_stack": ,
        "nums": ,
        "id": ,
        "original_text": ,
        "template": ,
    }
    """
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(train_fold, test_fold, 5, generate_nums,
                                                                    copy_nums, args.bert_path, args.max_seq_length, tree=True)
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    # pp = pprint.PrettyPrinter(indent=4)
    # print(train_pairs[100])
    # print([key for key in train_pairs[100]])



    # Initialize models
    encoder = EncoderSeq_OnlyBert(seq_length = args.max_seq_length, bert_path=args.bert_path)
    predict = Prediction(hidden_size=args.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=args.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=args.embedding_size)
    merge = Merge(hidden_size=args.hidden_size, embedding_size=args.embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings


    param_optimizer = list(encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_bert},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    encoder_optimizer = AdamW(optimizer_grouped_parameters,
                              lr = args.learning_rate_bert, # args.learning_rate - default is 5e-5
                              eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                              correct_bias = False
                              )
    encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer,
                                                        num_warmup_steps = int(train_steps * args.warmup_proportion), # Default value in run_glue.py
                                                        num_training_steps = train_steps)


    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate_bert, weight_decay=args.weight_decay_bert)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=25, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=args.step_size, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=args.step_size, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=args.step_size, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()



    best_equ_ac = 0
    best_val_ac = 0
    for epoch in range(args.n_epochs):
        random.seed(epoch + args.seed)

        loss_total = 0

        start = time.time()
        train_data = prepare_data_Batch(train_pairs, args.batch_size)
        print("Data Loading time", time_since(time.time() - start))

        start = time.time()

        print("epoch:", epoch + 1)
        # epoch_batch_count = 0
        num_accurate = 0
        for batch in tqdm(train_data):
            # max_token_len = batch["max_token_len"]
            loss, accurate = train_tree(batch["output"], batch["output_len"],
                                        batch["num_stack"], batch["num_size"], generate_nums,
                                        encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler,
                                        predict_optimizer, generate_optimizer,
                                        merge_optimizer, output_lang, batch["num_idx"],
                                        batch["token_ids_MaskN"] if args.MaskN else batch["token_ids"],
                                        batch["token_type_ids"],
                                        batch["attention_mask"])

            loss_total += loss
            num_accurate += accurate

        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()

        print("loss:", loss_total / len(train_data), "  accurate:", num_accurate/len(train_pairs))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")

        log_writer.add_scalar('Finetune/train/loss', loss_total/len(train_data), epoch)
        log_writer.add_scalar('Finetune/train/accurate', num_accurate/len(train_pairs), epoch)

        if epoch % 1 == 0 or epoch > args.n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            """
            data格式:
            {
                "token_ids":token_ids
                "token_type_ids":token_type_ids
                "attention_mask":attention_mask
                "output":output_cell
                "num_stack":num_stack
                "nums":pair["nums"]
                "original_text": text

                "tokens":tokens,
                "num_idx":num_idx,
                if MaskN:
                    "noun":[[1], [2,3]],
                    "name":[[3], [6,8]],
            }
            """
            for test_batch in tqdm(test_pairs):

                Noun = [2, 3, 4, 5, 6]
                Name = [7, 8]
                random.shuffle(Noun)
                random.shuffle(Name)

                if args.MaskN:
                    token_ids_MaskN = copy.deepcopy(test_batch["token_ids"])
                    for i in range(len(test_batch["noun"])):
                        for j in test_batch["noun"][i]:
                            token_ids_MaskN[j] = Noun[i]
                    for i in range(len(test_batch["name"])):
                        for j in test_batch["name"][i]:
                            token_ids_MaskN[j] = Name[i]

                token_len = len(test_batch["tokens"])
                test_res = evaluate_tree(generate_num_ids, encoder, predict, generate, merge,
                                         output_lang, test_batch["num_idx"],
                                         [token_ids_MaskN[:token_len]] if args.MaskN else [test_batch["token_ids"][:token_len]],
                                         [test_batch["token_type_ids"][:token_len]],
                                         [test_batch["attention_mask"][:token_len]],
                                         token_len, beam_size = args.beam_size)
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch["output"],
                                                                  output_lang, test_batch["nums"], test_batch["num_stack"])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            best_equ_ac = max(best_equ_ac, float(equation_ac) / eval_total)
            best_val_ac = max(best_val_ac, float(value_ac) / eval_total)
            print("Best_answer_acc", best_equ_ac, best_val_ac)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")

            if best_val_ac == float(value_ac)/eval_total :
                encoder.savebert(args.save_path + "/pytorch_model.bin")
                torch.save(predict.state_dict(), "%s/predict" % (args.save_path))
                torch.save(generate.state_dict(), "%s/generate" % (args.save_path))
                torch.save(merge.state_dict(), "%s/merge" % (args.save_path))

            log_writer.add_scalar('Finetune/test/accurate', float(value_ac)/eval_total, epoch)
