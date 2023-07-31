# coding: utf-8
import os
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

def set_args():
    parser = argparse.ArgumentParser(description = "bert2tree")
    parser.add_argument('--n_epochs', type=int, default=60)
    # parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--max_seq_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss_weight', nargs='+', type=float, default=[0.0, 0.15, 1, 2000])  # 0.15, 0.15, 1, 2000
    # parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--learning_rate_bert', type=float, default=1e-4)
    parser.add_argument('--weight_decay_bert', type=float, default=0.1)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--debug', action='store_true', default=False)

    tmp = parser.parse_args()
    parser.add_argument('--save_path', type=str, default="model_CRD_4loss_lr%.0e_ep%d_wd%.0e_lrb%.0e_wdb%.0e" 
        %(tmp.learning_rate, tmp.n_epochs, tmp.weight_decay, tmp.learning_rate_bert, tmp.weight_decay_bert))
    parser.add_argument('--ori_path', type=str, default='./data/')
    parser.add_argument('--prefix', type=str, default='23k_processed.json')
    parser.add_argument('--raw_data_path', type=str, default="data/Math_23K.json")
    parser.add_argument('--our_data_path', type=str, default="data/Math_23K_Noun_template_no_simplify.json")
    parser.add_argument('--student_path', type=str, default="./pretrained_lm/chinese-roberta-wwm-ext" )
    parser.add_argument('--teacher_path', type=str, default="teacher_model/")

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
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("make dir ", args.save_path)


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
                                                                    copy_nums, args.student_path, args.max_seq_length, tree=True)
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    # pp = pprint.PrettyPrinter(indent=4)
    # print(train_pairs[100])
    print([key for key in train_pairs[100]])

    # Initialize models
    student = Pretrain_Bert_CRD(bert_path=args.student_path)
    student_k = Pretrain_Bert_CRD(bert_path=args.student_path)
    teacher = Pretrain_Bert_CRD(bert_path=args.teacher_path)
    student_cls = Bert_classification_head(hidden_size=student.hidden_size)


    param_optimizer = list(student.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_bert},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    student_optimizer = AdamW(optimizer_grouped_parameters,
                    lr = args.learning_rate_bert, # args.learning_rate - default is 5e-5
                    eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                    correct_bias = False
                    )
    student_scheduler = get_linear_schedule_with_warmup(student_optimizer, 
                                        num_warmup_steps = int(train_steps * args.warmup_proportion), # Default value in run_glue.py
                                        num_training_steps = train_steps)

    student_cls_optimizer = torch.optim.Adam(student_cls.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    student_cls_scheduler = torch.optim.lr_scheduler.StepLR(student_cls_optimizer, step_size=25, gamma=0.5)

    print(len(train_pairs))

    myMoCo = MoCo_CRD(ori_len=len(train_pairs), K = 100 if args.debug else 10000)

    # # Move models to GPU
    if USE_CUDA:
        student.cuda()
        student_k.cuda()
        teacher.cuda()
        student_cls.cuda()

        # student_head.cuda()
        # teacher_head.cuda()

    train_data = prepare_data_Batch(train_pairs, args.batch_size)
    myMoCo.initialize_queue(student, teacher, train_data)

    # 在使用完teacher后就应该抛弃, 减少显存消耗
    # del teacher
    # torch.cuda.empty_cache()

    for epoch in range(args.n_epochs):
        random.seed(epoch + args.seed) 

        start = time.time()
        
        print(len(train_pairs))
        train_data = prepare_data_Batch(train_pairs, args.batch_size)
        
        print("epoch:", epoch + 1)
        total_loss = 0
        total_cls_loss = 0
        total_con_loss = 0
        total_crd_loss = 0
        total_distill_loss = 0

        pbar = tqdm(train_data)
        for batch in pbar:
            loss = pretrain_CRD(myMoCo, student, teacher, student_k, student_cls, student_optimizer, 
                student_scheduler, student_cls_optimizer, batch, loss_weight=args.loss_weight)

            total_loss += loss["loss"]
            total_cls_loss += loss["cls_loss"]
            total_con_loss += loss["con_loss"]
            total_crd_loss += loss["crd_loss"]
            total_distill_loss += loss["distill_loss"]

            pbar.set_description("## Loss: %f %f %f %f %f"%(loss["loss"], loss["cls_loss"], loss["con_loss"], loss["crd_loss"], loss["distill_loss"]))
            
        student_cls_scheduler.step()

        myMoCo.momentum_update_queue()
        
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

        if epoch%4 == 0:
            student.savebert(args.save_path + "/pytorch_model_epoch" + str(epoch) + "_" + str(total_loss) + ".bin")
