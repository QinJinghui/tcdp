from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        test = out_expression_list(test_res, output_lang, num_list)
        tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
        # return True, True, test_res, test_tar
        return True, True, test, tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    # masked_index = torch.ByteTensor(masked_index)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # B x S x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # B x S x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)




def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

def pretrain_contrastive(encoder, encoder_optimizer, encoder_scheduler, 
               token_ids, token_ids_, token_type_ids, attention_mask, templates, criterion, english=False):
    encoder.train()
    if USE_CUDA:
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_ids_ = torch.tensor(token_ids_, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        templates = torch.tensor(templates, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    encoder_outputs_, problem_output_ = encoder(token_ids_, token_type_ids, attention_mask)
    
    problem_output = torch.cat([problem_output.unsqueeze(1), problem_output_.unsqueeze(1)], dim=1)
    loss = criterion(problem_output, templates)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    return loss.item()

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
def pretrain_CRD(myMoCo, student, teacher, student_k, student_cls, student_optimizer, student_scheduler, student_cls_optimizer, batch, loss_weight):
    student.train()
    student_cls.train()
    student_k.eval()
    teacher.eval()

    if USE_CUDA:
        token_ids = torch.tensor(batch["token_ids"], dtype=torch.long).cuda()
        token_ids_MaskN = torch.tensor(batch["token_ids_MaskN"], dtype=torch.long).cuda()
        # token_ids_DisturbN = torch.tensor(batch["token_ids_DisturbN"], dtype=torch.long).cuda()
        token_type_ids = torch.tensor(batch["token_type_ids"], dtype=torch.long).cuda()
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
        templates = torch.tensor(batch["template"], dtype=torch.long).cuda()
    
    student_optimizer.zero_grad()
    student_cls_optimizer.zero_grad()

    hidden_t, hidden_mean_t, cls_t, _, _ = teacher(token_ids, token_type_ids, attention_mask)
    hidden_t = hidden_t.detach()
    hidden_mean_t = hidden_mean_t.detach()
    cls_t = cls_t.detach()

    _, _, _, hidden_maskN, _ = student_k(token_ids_MaskN, token_type_ids, attention_mask)
    hidden_maskN = hidden_maskN.detach()
    
    # _, _, _, hidden_disturbN, _ = student_k(token_ids_DisturbN, token_type_ids, attention_mask)
    # hidden_disturbN = hidden_disturbN.detach()

    hidden_s, hidden_mean_s, cls_s, hidden_mlp, _ = student(token_ids, token_type_ids, attention_mask)
    cls_loss = student_cls(hidden_mlp, templates)

    con_loss = myMoCo.contrastive_loss(hidden_mlp, hidden_maskN, templates)
    # crd_loss = myMoCo.crd_loss(hidden_mean_s, templates)
    crd_loss = myMoCo.crd_loss(hidden_mlp, templates)
    distill_loss = myMoCo.distill_loss(hidden_s, cls_s, hidden_t, cls_t)

    loss = loss_weight[0]*cls_loss + loss_weight[1]*con_loss + loss_weight[2]*crd_loss + loss_weight[3]*distill_loss   
    loss.backward()

    student_optimizer.step()
    student_scheduler.step()
    student_cls_optimizer.step()

    myMoCo.momentum_update_model(student, student_k)

    myMoCo.push_queue_maskN(hidden_maskN)
    # myMoCo.push_queue_disturbN(hidden_disturbN)
    myMoCo.push_label(templates.detach())

    myMoCo.pop_queue()
    
    assert len(myMoCo.queue_maskN) == myMoCo.K 
    # assert len(myMoCo.queue_disturbN) == myMoCo.K 
    assert len(myMoCo.label) == myMoCo.K

    loss_dict = {
        "loss": loss.item(),
        "cls_loss": cls_loss.item(),
        "con_loss": con_loss.item(), 
        "crd_loss": crd_loss.item(), 
        "distill_loss": distill_loss.item()
    }

    return loss_dict


def pretrain_my_CRD(myMoCo, student, teacher, student_k, student_cls, student_optimizer, student_scheduler, student_cls_optimizer, batch, loss_weight):
    student.train()
    student_cls.train()
    student_k.eval()
    teacher.eval()

    if USE_CUDA:
        token_ids = torch.tensor(batch["token_ids"], dtype=torch.long).cuda()
        token_ids_MaskN = torch.tensor(batch["token_ids_MaskN"], dtype=torch.long).cuda()
        # token_ids_DisturbN = torch.tensor(batch["token_ids_DisturbN"], dtype=torch.long).cuda()
        token_type_ids = torch.tensor(batch["token_type_ids"], dtype=torch.long).cuda()
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
        templates = torch.tensor(batch["template"], dtype=torch.long).cuda()

    student_optimizer.zero_grad()
    student_cls_optimizer.zero_grad()

    # last_hidden_state, hidden_mean, pooler_output, hidden_mean_mlp, pooler_output_mlp
    hidden_t, hidden_mean_t, cls_t, _, _ = teacher(token_ids, token_type_ids, attention_mask)
    hidden_t = hidden_t.detach()
    hidden_mean_t = hidden_mean_t.detach()
    cls_t = cls_t.detach()

    # last_hidden_state, hidden_mean, pooler_output, hidden_mean_mlp, pooler_output_mlp
    # _, _, cls_k, hidden_maskN, cls_k_mlp = student_k(token_ids_MaskN, token_type_ids, attention_mask)
    # hidden_k, hidden_mean_k, cls_k, _, _ = student_k(token_ids_MaskN, token_type_ids, attention_mask)
    hidden_k, hidden_mean_k, cls_k, _, _ = student_k(token_ids, token_type_ids, attention_mask)
    # hidden_maskN = hidden_maskN.detach()
    hidden_k = hidden_k.detach()
    hidden_mean_k = hidden_mean_k.detach()
    cls_k = cls_k.detach()
    # cls_k_mlp = cls_k_mlp.detach()

    # _, _, _, hidden_disturbN, _ = student_k(token_ids_DisturbN, token_type_ids, attention_mask)
    # hidden_disturbN = hidden_disturbN.detach()

    hidden_s, hidden_mean_s, cls_s, hidden_mlp, cls_s_mlp = student(token_ids, token_type_ids, attention_mask)
    cls_loss = student_cls(hidden_mlp, templates)

    # hidden_maskN = hidden_maskN.detach()
    # con_loss = myMoCo.contrastive_loss(hidden_mlp, hidden_maskN, templates)
    # con_loss = myMoCo.contrastive_loss(cls_s, cls_k, templates)
    con_loss = myMoCo.contrastive_loss(hidden_mean_s, hidden_mean_k, templates)
    # crd_loss = myMoCo.crd_loss(hidden_mean_s, templates)
    crd_loss = myMoCo.crd_loss(cls_s, templates)
    distill_loss = myMoCo.distill_loss(hidden_s, cls_s, hidden_t, cls_t)

    loss = loss_weight[1]*con_loss + loss_weight[2]*crd_loss + loss_weight[3]*distill_loss  # loss_weight[0]*cls_loss+
    loss.backward()

    student_optimizer.step()
    student_scheduler.step()
    student_cls_optimizer.step()

    myMoCo.momentum_update_model()

    # myMoCo.push_moco_queue(hidden_maskN, templates.detach())
    # myMoCo.push_student_queue(hidden_maskN, templates.detach())
    myMoCo.push_moco_queue(hidden_mean_s.detach(), templates.detach())
    myMoCo.push_student_queue(cls_s.detach(), templates.detach())
    # myMoCo.push_queue_disturbN(hidden_disturbN)
    # myMoCo.push_label(templates.detach())

    myMoCo.pop_moco_queue(token_ids.size(0))
    myMoCo.pop_student_queue(token_ids.size(0))

    assert len(myMoCo.student_queue) == myMoCo.K
    # assert len(myMoCo.queue_disturbN) == myMoCo.K
    assert len(myMoCo.label) == myMoCo.K

    loss_dict = {
        "loss": loss.item(),
        "cls_loss": cls_loss.item(),
        "con_loss": con_loss.item(),
        "crd_loss": crd_loss.item(),
        "distill_loss": distill_loss.item()
    }

    return loss_dict


def pretrain_contrastive_moco(myMoCo, encoder_q, encoder_k, encoder_optimizer, encoder_scheduler,
               token_ids_q, token_ids_k, token_type_ids, attention_mask, templates, english=False):
    encoder_q.train()
    encoder_k.eval()
    if USE_CUDA:
        token_ids_q = torch.tensor(token_ids_q, dtype=torch.long).cuda()
        token_ids_k = torch.tensor(token_ids_k, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        templates = torch.tensor(templates, dtype=torch.long).cuda()

    encoder_optimizer.zero_grad()

    encoder_outputs_q, problem_output_q = encoder_q(token_ids_q, token_type_ids, attention_mask)
    encoder_outputs_k, problem_output_k = encoder_k(token_ids_k, token_type_ids, attention_mask)
    problem_output_k = problem_output_k.detach()
    
    loss = myMoCo.contrastive_loss(problem_output_q, problem_output_k, templates)

    loss.backward()
    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()

    myMoCo.momentum_update(encoder_q, encoder_k)
    myMoCo.push_queue(problem_output_k, templates.detach())
    myMoCo.pop_queue()
    return loss.item()


def train_tree(output, output_len, nums_stack, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask, english=False):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    target = torch.LongTensor(output).transpose(0, 1)

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        # input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_idx]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), output_len)
    loss, accurate = masked_cross_entropy(all_node_outputs, target, output_len)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree(generate_nums, encoder, predict, generate, merge, output_lang, 
                  num_pos, token_ids, token_type_ids, attention_mask, input_len_max, 
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(attention_mask)
    seq_mask = torch.BoolTensor(1, input_len_max).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        # input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out

