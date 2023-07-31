import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
USE_CUDA = torch.cuda.is_available()

class MoCo(nn.Module):
    def __init__(self, hidden_size=768, K=10000, beta=0.999, T=0.1):
        self.hidden_size = hidden_size
        self.K = K
        self.beta = beta
        self.T = T

        self.queue = torch.zeros((0, self.hidden_size), dtype=torch.float)  # tensor([])
        self.label = torch.zeros(0)  # tensor([])
        if USE_CUDA:
            self.queue = self.queue.cuda()
            self.label = self.label.cuda()
        self.queue_initialized = False

    def push_queue(self, data, label):
        self.queue = torch.cat([self.queue, data], dim=0)
        self.label = torch.cat([self.label, label], dim=0)
    
    def pop_queue(self):
        self.queue = self.queue[-self.K:]
        self.label = self.label[-self.K:]

    def initialize_queue(self, model_k, token_ids_batches_, 
        token_type_ids_batches, attention_mask_batches, template_batches):
        if self.queue_initialized:
            return 
        print("## Initializing queue...")
        model_k.eval()
        pbar = tqdm(range(len(token_ids_batches_)))
        for idx in pbar:
            token_ids_k = token_ids_batches_[idx]
            token_type_ids = token_type_ids_batches[idx]
            attention_mask = attention_mask_batches[idx]
            template = template_batches[idx]
            if USE_CUDA:
                token_ids_k = torch.tensor(token_ids_k, dtype=torch.long).cuda()
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
                template = torch.tensor(template, dtype=torch.long).cuda()

            encoder_outputs_k, problem_output_k = model_k(token_ids_k, token_type_ids, attention_mask)
            
            problem_output_k = problem_output_k.detach()
            self.push_queue(problem_output_k, template)
            if len(self.queue) > self.K:
                self.pop_queue()
        self.queue_initialized = True

        print("## queue len: ", len(self.queue))
        print("## label len: ", len(self.label))
    
    def momentum_update(self, model_q, model_k):
        """ model_k = m * model_k + (1 - m) model_q """
        for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
            p2.data.mul_(self.beta).add_(p1.detach().data, alpha=1-self.beta)

    def get_shuffle_idx(self, bs):
        """shuffle index for ShuffleBN """
        shuffle_value = torch.randperm(bs).long()  # index 2 value
        reverse_idx = torch.zeros(bs).long()
        arange_index = torch.arange(bs).long()

        if USE_CUDA:
            shuffle_value = shuffle_value.cuda()
            reverse_idx = reverse_idx.cuda()
            arange_index = arange_index.cuda()

        reverse_idx.index_copy_(0, shuffle_value, arange_index)  # value back to index
        return shuffle_value, reverse_idx

    def contrastive_loss(self, feature_q, feature_k, template):
        # feature: [B x hidden_size]
        # template: [B]
        N, H = feature_q.shape 

        mask = torch.eq(template.view(N,1).repeat(1, len(self.queue)), self.label.repeat(N, 1)).float() # [N x K]

        logits_qk = torch.bmm(feature_q.view(N, 1, H), feature_k.view(N, H, 1)).view(N,1)
        logits_q_queue = torch.mm(feature_q.view(N, H), self.queue.transpose(0, 1)) # [N x K]

        logits = torch.cat([logits_q_queue, logits_qk], dim=1)/self.T # [N x K+1]
        mask_add = torch.ones(N, 1)
        if USE_CUDA:
            mask_add = mask_add.cuda()
        mask_all = torch.cat([mask, mask_add], dim=1) # [N x K+1]
        num_positive_row = mask_all.sum(1, keepdim=True)

        exp_log = torch.exp(logits) # [N x K+1]
        exp_log_sum = torch.sum(exp_log, dim=1, keepdim=True) # [N, 1]
        exp_log = torch.log(exp_log/exp_log_sum) # [N x K+1]

        log_pi_sum = torch.sum(exp_log*mask_all, dim=1, keepdim=True)/num_positive_row # [N*1]
        
        loss = - torch.sum(log_pi_sum)
        return loss
    

class MoCo_CRD(nn.Module):
    def __init__(self, ori_len=21162, hidden_size=768, K=1500, beta=0.999, M=200, T=0.1, momentum=0.5):
        self.hidden_size = hidden_size
        self.K = K
        self.beta = beta
        self.M = M
        self.T = T
        self.momentum = momentum

        self.queue_teacher = F.normalize(torch.rand(ori_len, self.hidden_size, dtype=torch.float))
        self.queue_teacher_fixed = torch.zeros((0, self.hidden_size), dtype=torch.float)
        self.label_teacher = torch.zeros(0, dtype=torch.long)

        self.queue_maskN = torch.zeros((0, self.hidden_size), dtype=torch.float)
        # self.queue_disturbN = torch.zeros((0, self.hidden_size), dtype=torch.float)

        self.label = torch.zeros(0, dtype=torch.long)

        if USE_CUDA:
            self.queue_teacher = self.queue_teacher.cuda()
            self.queue_teacher_fixed = self.queue_teacher_fixed.cuda()
            self.label_teacher = self.label_teacher.cuda()

            self.queue_maskN = self.queue_maskN.cuda()
            # self.queue_disturbN = self.queue_disturbN.cuda()
            self.label = self.label.cuda()

        self.queue_initialized = False

    def push_queue_teacher(self, data, label_teacher):
        self.queue_teacher_fixed = torch.cat([self.queue_teacher_fixed, data], dim=0)
        self.label_teacher = torch.cat([self.label_teacher, label_teacher])
    
    def push_queue_maskN(self, data):
        self.queue_maskN = torch.cat([self.queue_maskN, data], dim=0)
    
    # def push_queue_disturbN(self, data):
    #     self.queue_disturbN = torch.cat([self.queue_disturbN, data], dim=0)
    
    def push_label(self, label):
        self.label = torch.cat([self.label, label], dim=0)

    def momentum_update_model(self, model_q, model_k):
        """ model_k = m * model_k + (1 - m) model_q """
        for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
            p2.data.mul_(self.beta).add_(p1.detach().data, alpha=1-self.beta)
    
    def momentum_update_queue(self):
        self.queue_teacher.mul_(self.momentum)
        self.queue_teacher.add_(torch.mul(self.queue_teacher_fixed, 1 - self.momentum))
        self.queue_teacher = F.normalize(self.queue_teacher)

    def pop_queue(self):
        self.queue_maskN = self.queue_maskN[-self.K:]
        # self.queue_disturbN = self.queue_disturbN[-self.K:]
        self.label = self.label[-self.K:]

    def initialize_queue(self, student, teacher, data):
        if self.queue_initialized:
            print("queue has benn initialized...")
            return True

        print("## Initializing queue...")
        student.eval()
        teacher.eval()

        pbar = tqdm(range(len(data)))
        for idx in pbar:
            batch = data[idx]

            if USE_CUDA:
                token_ids = torch.tensor(batch["token_ids"], dtype=torch.long).cuda()
                token_ids_MaskN = torch.tensor(batch["token_ids_MaskN"], dtype=torch.long).cuda()
                # token_ids_DisturbN = torch.tensor(batch["token_ids_DisturbN"], dtype=torch.long).cuda()
                token_type_ids = torch.tensor(batch["token_type_ids"], dtype=torch.long).cuda()
                attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
                label = torch.tensor(batch["template"], dtype=torch.long).cuda()

            # _, output_ori, _, _, _ = teacher(token_ids, token_type_ids, attention_mask)
            # change to
            _, _, _, output_ori, _ = teacher(token_ids, token_type_ids, attention_mask)

            # _, _, _, output_maskN, _ = student(token_ids_MaskN, token_type_ids, attention_mask) # orig
            _, _, _, output_maskN, _ = student(token_ids, token_type_ids, attention_mask)
            # _, _, _, output_disturbN, _ = student(token_ids_DisturbN, token_type_ids, attention_mask)
            
            output_ori = output_ori.detach()
            output_maskN = output_maskN.detach()
            # output_disturbN = output_disturbN.detach()

            self.push_queue_teacher(output_ori, label)
            self.push_queue_maskN(output_maskN)
            # self.push_queue_disturbN(output_disturbN)
            self.push_label(label)

            if len(self.label) > self.K:
                self.pop_queue()

        assert len(self.queue_maskN) == len(self.label)
        # assert len(self.queue_disturbN) == len(self.label)
        self.queue_initialized = True

        print("## CRD len: ", len(self.queue_teacher))
        print("## moco len: ", len(self.queue_maskN))
        print("## label len: ", len(self.label))

        return False
    
    def contrastive_loss(self, feature_q, feature_k_maskN, template):
        # feature: [B x hidden_size]
        # template: [B]
        N, H = feature_q.shape
        print("feature_q:", feature_q.size())
        print("feature_k_maskN:", feature_k_maskN.size())
        print("template:", template.size())
        print("queue_maskN:", self.queue_maskN.size())
        print("label:", self.label.size())
        mask = torch.eq(template.view(N,1).repeat(1, len(self.queue_maskN)), self.label.repeat(N, 1)).float() # [N x K]
        print("mask:", mask.size())
        print(mask)

        logits_qk = torch.bmm(feature_q.view(N, 1, H), feature_k_maskN.view(N, H, 1)).view(N,1)
        logits_q_queue = torch.mm(feature_q.view(N, H), self.queue_maskN.transpose(0, 1)) # [N x K]

        logits = torch.cat([logits_q_queue, logits_qk], dim=1)/self.T # [N x K+1]
        mask_add = torch.ones(N, 1)
        if USE_CUDA:
            mask_add = mask_add.cuda()
        mask_all = torch.cat([mask, mask_add], dim=1) # [N x K+1]
        num_positive_row = mask_all.sum(1, keepdim=True)

        exp_log = torch.exp(logits) # [N x K+1]
        exp_log_sum = torch.sum(exp_log, dim=1, keepdim=True) # [N, 1]
        exp_log = torch.log(exp_log/exp_log_sum) # [N x K+1]

        log_pi_sum = torch.sum(exp_log*mask_all, dim=1, keepdim=True)/num_positive_row # [N*1]
        
        loss = - torch.mean(log_pi_sum)
        return loss

    def crd_loss(self, feature, template):
        # feature: [B x hidden_size]
        # template: [B]

        N, H = feature.shape 
        mask_positive = torch.eq(template.view(N,1).repeat(1, len(self.queue_teacher)), self.label_teacher.repeat(N, 1)).float() # [N x train_len]
        mask_negative = 1 - mask_positive

        num_positive_row = mask_positive.sum(1)
        num_negative_row = mask_negative.sum(1)

        assert not 0 in num_positive_row 
        assert not 0 in num_negative_row

        exp_dot_st = torch.exp(torch.mm(feature.view(N, H), self.queue_teacher.transpose(0, 1))/self.T) # [N x train_len]
        h_ts = exp_dot_st / (exp_dot_st + self.M)

        log_h_positive = torch.log(h_ts)
        log_h_negative = torch.log(1 - h_ts)

        positive_loss = torch.sum(log_h_positive * mask_positive, dim=1) / num_positive_row
        negative_loss = torch.sum(log_h_negative * mask_negative, dim=1) / num_negative_row

        loss = - torch.mean(positive_loss + negative_loss)
        return loss
    
    def distill_loss(self, hidden_s, cls_s, hidden_t, cls_t):
        B = hidden_s.shape[0]
        hidden_s = hidden_s.reshape(B, -1)
        hidden_t = hidden_t.reshape(B, -1)

        loss_fct = nn.MSELoss()

        cls_los = loss_fct(cls_s, cls_t)
        hidden_loss = loss_fct(hidden_s, hidden_t)
        return cls_los + hidden_loss


class MySupervisedMoCoCRD(nn.Module):
    def __init__(self, student_model, student_k_model, teacher_model,
                 ori_len=21162, hidden_size=768, K=1500, m=0.999, T=0.07,  #T=0.07
                 crd_momentum=0.5, beta=0.999, M=200,):
        super(MySupervisedMoCoCRD, self).__init__()
        self.hidden_size = hidden_size
        self.K = K  # K: queue size; number of negative keys (default: 65536)
        self.m = m  # m: moco momentum of updating key encoder (default: 0.999)
        self.T = T  # T: softmax temperature (default: 0.07)
        self.cross_entopy_loss = nn.BCEWithLogitsLoss()  #nn.CrossEntropyLoss()
        self.aug = nn.Dropout(0.2)

        self.crd_momentum = crd_momentum
        self.M = M
        self.beta = beta

        self.student_model_q = student_model
        self.student_model_k = student_k_model

        self.teacher_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.student_q_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.student_k_fc = nn.Linear(self.hidden_size, self.hidden_size)
        if USE_CUDA:
            self.teacher_fc = self.teacher_fc.cuda()
            self.student_q_fc = self.student_q_fc.cuda()
            self.student_k_fc = self.student_k_fc.cuda()
            self.relu = self.relu.cuda()

        for param_q, param_k in zip(self.student_model_q.parameters(), self.student_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.student_q_fc.parameters(), self.student_k_fc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.moco_queue = torch.zeros((0, self.hidden_size), dtype=torch.float)  # tensor([])
        self.label = torch.zeros(0)  # tensor([])
        if USE_CUDA:
            self.moco_queue = self.moco_queue.cuda()
            self.label = self.label.cuda()
        self.moco_queue_initialized = False
        self.moco_queue_ptr = torch.zeros(1, dtype=torch.long)


        # CRD
        self.teacher_model = teacher_model
        stdv = 1. / math.sqrt(ori_len / 3)
        self.crd_params = torch.tensor([K, T, -1, -1, self.crd_momentum])
        self.student_queue = torch.zeros((0, self.hidden_size), dtype=torch.float) # torch.rand(ori_len, self.hidden_size, dtype=torch.float).mul_(2 * stdv).add_(-stdv)
        self.teacher_queue = torch.zeros((0, self.hidden_size), dtype=torch.float) # torch.rand(ori_len, self.hidden_size, dtype=torch.float).mul_(2 * stdv).add_(-stdv)
        self.teacher_queue_fixed = torch.zeros((0, self.hidden_size), dtype=torch.float)
        self.teacher_label = torch.zeros(0, dtype=torch.long)
        self.student_label = torch.zeros(0, dtype=torch.long)

        self.crd_queue_initialized = False

        if USE_CUDA:
            self.student_queue = self.student_queue.cuda()
            self.teacher_queue = self.teacher_queue.cuda()
            self.teacher_queue_fixed = self.teacher_queue_fixed.cuda()
            self.teacher_label = self.teacher_label.cuda()
            self.student_label = self.student_label.cuda()

        self.dist_loss = nn.MSELoss()

    @torch.no_grad()
    def momentum_update_model(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.student_model_q.parameters(), self.student_model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.student_q_fc.parameters(), self.student_k_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def initialize_moco_queue(self, data):
        # token_ids_batches_, token_type_ids_batches,
        #                          attention_mask_batches, template_batches
        if self.moco_queue_initialized:
            return
        print("## Initializing MoCo queue...")
        self.student_model_k.eval()
        pbar = tqdm(range(len(data)))
        for idx in pbar:
            batch = data[idx]
            token_ids_k = batch["token_ids"]
            token_type_ids = batch["token_type_ids"]
            attention_mask = batch["attention_mask"]
            template = batch["template"]
            if USE_CUDA:
                token_ids_k = torch.tensor(token_ids_k, dtype=torch.long).cuda()
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
                template = torch.tensor(template, dtype=torch.long).cuda()
            else:
                token_ids_k = torch.tensor(token_ids_k, dtype=torch.long)
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                template = torch.tensor(template, dtype=torch.long)

            # last_hidden_state, hidden_mean, pooler_output, hidden_mean_mlp, pooler_output_mlp
            # encoder_outputs_k, _, problem_output_k, _, _ = self.student_model_k(token_ids_k, token_type_ids, attention_mask)
            # encoder_outputs_k, _, _, problem_output_k, _ = self.student_model_k(token_ids_k, token_type_ids, attention_mask)
            last_hidden_state, hidden_mean, cls_k, hidden_mean_mlp, cls_k_mlp = self.student_model_k(token_ids_k, token_type_ids, attention_mask)
            hidden_mean = self.aug(hidden_mean)
            # cls_k = F.normalize(self.student_k_fc(self.relu(cls_k)), dim=1)
            cls_k = F.normalize(self.student_k_fc(self.relu(hidden_mean)), dim=1)

            cls_k = cls_k.detach()
            self.push_moco_queue(cls_k, template)

            # problem_output_k = problem_output_k.detach()
            # self.push_moco_queue(problem_output_k, template)
            if len(self.moco_queue) > self.K:
                self.moco_queue = self.moco_queue[-self.K:]
                self.label = self.label[-self.K:]
        self.moco_queue_initialized = True

        print("## queue len: ", len(self.moco_queue))
        print("## label len: ", len(self.label))

    def push_moco_queue(self, data, label):
        self.moco_queue = torch.cat([self.moco_queue, data], dim=0)
        self.label = torch.cat([self.label, label], dim=0)

    def pop_moco_queue(self, batch_size):
        self.moco_queue = self.moco_queue[batch_size:]
        self.label = self.label[batch_size:]

    def contrastive_loss(self, feature_q, feature_k, template):
        # feature: [B x hidden_size]
        # template: [B]
        # feature_q = self.aug(feature_q)
        feature_k = self.aug(feature_k)
        feature_q = F.normalize(self.student_q_fc(self.relu(feature_q)), dim=1)
        feature_k = F.normalize(self.student_k_fc(self.relu(feature_k)), dim=1)
        feature_k = feature_k.detach()

        N, H = feature_q.shape
        # print("template:", template.size())
        # print("queue_maskN:", self.moco_queue.size())
        # print("label:", self.label.size())
        mask = torch.eq(template.view(N, 1).repeat(1, len(self.moco_queue)), self.label.repeat(N, 1)).float() # [N x K]
        # print("mask:", mask.size())
        # print(mask)

        logits_pos = torch.bmm(feature_q.view(N, 1, H), feature_k.view(N, H, 1)).view(N,1) # [N x 1]
        logits_neg = torch.mm(feature_q.view(N, H), self.moco_queue.transpose(0, 1)) # [N x K]

        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.T # [N x K+1]
        mask_add = torch.ones(N, 1)
        if USE_CUDA:
            mask_add = mask_add.cuda()
        mask_all = torch.cat([mask_add, mask], dim=1)  # [N x K+1]
        num_positive_row = mask_all.sum(dim=1, keepdim=True)

        # exp_log = torch.exp(logits) # [N x K+1]
        # exp_log_sum = torch.sum(exp_log, dim=1, keepdim=True) # [N, 1]
        # exp_log = torch.log(exp_log/exp_log_sum) # [N x K+1]
        #
        # # change to BCE ?
        # log_pi_sum = torch.sum(exp_log*mask_all, dim=1, keepdim=True)/num_positive_row # [N*1]
        #
        # loss = - torch.mean(log_pi_sum)
        # return loss

        loss = self.cross_entopy_loss(logits.view(-1), mask_all.view(-1))
        return loss

    def initialize_crd_queue(self, data):
        if self.crd_queue_initialized:
            print("queue has benn initialized...")
            return True

        print("## Initializing CRD queue...")
        self.student_model_q.eval()
        self.teacher_model.eval()

        pbar = tqdm(range(len(data)))
        for idx in pbar:
            batch = data[idx]

            if USE_CUDA:
                token_ids = torch.tensor(batch["token_ids"], dtype=torch.long).cuda()
                # token_ids_MaskN = torch.tensor(batch["token_ids_MaskN"], dtype=torch.long).cuda()
                # token_ids_DisturbN = torch.tensor(batch["token_ids_DisturbN"], dtype=torch.long).cuda()
                token_type_ids = torch.tensor(batch["token_type_ids"], dtype=torch.long).cuda()
                attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
                label = torch.tensor(batch["template"], dtype=torch.long).cuda()

            # last_hidden_state, hidden_mean, pooler_output, hidden_mean_mlp, pooler_output_mlp
            # _, output_ori, _, _, _ = teacher(token_ids, token_type_ids, attention_mask)
            # change to
            # _, _, _, output_teacher, _ = self.teacher_model(token_ids, token_type_ids, attention_mask)
            hidden_t, hidden_mean_t, cls_t, _, _ = self.teacher_model(token_ids, token_type_ids, attention_mask)

            # _, _, _, output_maskN, _ = student(token_ids_MaskN, token_type_ids, attention_mask) # orig
            # _, _, _, output_student, _ = self.student_model_q(token_ids, token_type_ids, attention_mask)
            hidden_s, hidden_mean_s, cls_s, _, _ = self.student_model_q(token_ids, token_type_ids, attention_mask)
            # _, _, _, output_disturbN, _ = student(token_ids_DisturbN, token_type_ids, attention_mask)

            cls_t = F.normalize(self.teacher_fc(self.relu(cls_t)), dim=1)
            cls_s = F.normalize(self.student_q_fc(self.relu(cls_s)), dim=1)

            output_teacher = cls_t.detach()
            output_student = cls_s.detach()

            self.push_teacher_queue(output_teacher, label)
            self.push_student_queue(output_student, label)

            if len(self.student_label) > self.K:
                self.student_queue = self.student_queue[-self.K:]
                self.student_label = self.student_label[-self.K:]

        assert len(self.student_queue) == len(self.student_label)
        # assert len(self.queue_disturbN) == len(self.label)
        self.crd_queue_initialized = True

        print("## CRD teacher len: ", len(self.teacher_queue))
        print("## CRD student len: ", len(self.student_queue))
        print("## label len: ", len(self.label))

        return False

    def push_teacher_queue(self, data, label_teacher):
        self.teacher_queue_fixed = torch.cat([self.teacher_queue_fixed, data], dim=0)
        self.teacher_queue = torch.cat([self.teacher_queue, data], dim=0)
        self.teacher_label = torch.cat([self.teacher_label, label_teacher])

    def push_student_queue(self, data, label_student):
        self.student_queue = torch.cat([self.student_queue, data], dim=0)
        self.student_label = torch.cat([self.student_label, label_student])

    def pop_teacher_queue(self, batch_size):
        self.teacher_queue = self.teacher_queue[batch_size:]
        self.teacher_label = self.teacher_label[batch_size:]

    def pop_student_queue(self, batch_size):
        self.student_queue = self.student_queue[batch_size:]
        self.student_label = self.student_label[batch_size:]

    def momentum_update_queue(self):
        self.teacher_queue.mul_(self.crd_momentum)
        self.teacher_queue.add_(torch.mul(self.teacher_queue_fixed, 1 - self.crd_momentum))
        # self.teacher_queue = F.normalize(self.teacher_queue)

    def crd_loss(self, feature_student, template):
        feature_student = F.normalize(self.student_q_fc(self.relu(feature_student)), dim=1)
        N, H = feature_student.shape

        mask = torch.eq(template.view(N, 1).repeat(1, len(self.teacher_queue)), self.teacher_label.repeat(N, 1)).float() # [N x K]

        if USE_CUDA:
            mask = mask.cuda()

        logits = torch.mm(feature_student.view(N, H), self.teacher_queue.transpose(0, 1)) / self.T   # [N x K]

        loss = self.cross_entopy_loss(logits.view(-1), mask.view(-1))
        return loss

        # num_positive_row = mask_positive.sum(1)
        # num_negative_row = mask_negative.sum(1)

        # assert not 0 in num_positive_row
        # # assert not 0 in num_negative_row
        #
        #
        #
        # logits_pos = torch.bmm(feature_student.view(N, 1, H), feature_student.view(N, H, 1)).view(N, 1)  # [N x 1]
        #
        #
        # mask_positive = torch.eq(template.view(N,1).repeat(1, len(self.teacher_queue)), self.teacher_label.repeat(N, 1)).float() # [N x train_len]
        # mask_negative = 1 - mask_positive
        #
        # num_positive_row = mask_positive.sum(1)
        # num_negative_row = mask_negative.sum(1)
        #
        # assert not 0 in num_positive_row
        # assert not 0 in num_negative_row
        #
        # exp_dot_st = torch.exp(torch.mm(feature_student.view(N, H), self.teacher_queue.transpose(0, 1))/self.T) # [N x train_len]
        # h_ts = exp_dot_st / (exp_dot_st + self.M)
        #
        # log_h_positive = torch.log(h_ts)
        # log_h_negative = torch.log(1 - h_ts)
        #
        # positive_loss = torch.sum(log_h_positive * mask_positive, dim=1) / num_positive_row
        # negative_loss = torch.sum(log_h_negative * mask_negative, dim=1) / num_negative_row
        #
        # loss = - torch.mean(positive_loss + negative_loss)
        # return loss

    # def crd_loss(self, feature_student, template):
    #     N, H = feature_student.shape
    #     mask_positive = torch.eq(template.view(N,1).repeat(1, len(self.teacher_queue)), self.teacher_label.repeat(N, 1)).float() # [N x train_len]
    #     mask_negative = 1 - mask_positive
    #
    #     num_positive_row = mask_positive.sum(1)
    #     num_negative_row = mask_negative.sum(1)
    #
    #     assert not 0 in num_positive_row
    #     assert not 0 in num_negative_row
    #
    #     exp_dot_st = torch.exp(torch.mm(feature_student.view(N, H), self.teacher_queue.transpose(0, 1))/self.T) # [N x train_len]
    #     h_ts = exp_dot_st / (exp_dot_st + self.M)
    #
    #     log_h_positive = torch.log(h_ts)
    #     log_h_negative = torch.log(1 - h_ts)
    #
    #     positive_loss = torch.sum(log_h_positive * mask_positive, dim=1) / num_positive_row
    #     negative_loss = torch.sum(log_h_negative * mask_negative, dim=1) / num_negative_row
    #
    #     loss = - torch.mean(positive_loss + negative_loss)
    #     return loss

    def distill_loss(self, hidden_s, cls_s, hidden_t, cls_t):
        B = hidden_s.shape[0]
        hidden_s = hidden_s.reshape(B, -1)
        hidden_t = hidden_t.reshape(B, -1)

        cls_los = self.dist_loss(cls_s, cls_t)
        hidden_loss = self.dist_loss(hidden_s, hidden_t)
        return cls_los + hidden_loss

        # loss_fct = nn.MSELoss()
        #
        # cls_los = loss_fct(cls_s, cls_t)
        # hidden_loss = loss_fct(hidden_s, hidden_t)
        # return cls_los + hidden_loss


    # def push_teacher_label(self, label):
    #     self.label = torch.cat([self.label, label], dim=0)



    # def push_queue_teacher(self, data, label_teacher):
    #     self.queue_teacher_fixed = torch.cat([self.queue_teacher_fixed, data], dim=0)
    #     self.label_teacher = torch.cat([self.label_teacher, label_teacher])
    #
    # def push_queue_maskN(self, data):
    #     self.queue_maskN = torch.cat([self.queue_maskN, data], dim=0)
    #
    # # def push_queue_disturbN(self, data):
    # #     self.queue_disturbN = torch.cat([self.queue_disturbN, data], dim=0)
    #
    # def push_label(self, label):
    #     self.label = torch.cat([self.label, label], dim=0)
    #
    # def momentum_update_model(self, model_q, model_k):
    #     """ model_k = m * model_k + (1 - m) model_q """
    #     for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
    #         p2.data.mul_(self.beta).add_(p1.detach().data, alpha=1-self.beta)
    #
    # def momentum_update_queue(self):
    #     self.queue_teacher.mul_(self.momentum)
    #     self.queue_teacher.add_(torch.mul(self.queue_teacher_fixed, 1 - self.momentum))
    #     self.queue_teacher = F.normalize(self.queue_teacher)
    #
    # def pop_queue(self):
    #     self.queue_maskN = self.queue_maskN[-self.K:]
    #     # self.queue_disturbN = self.queue_disturbN[-self.K:]
    #     self.label = self.label[-self.K:]
    #
    # def initialize_queue(self, student, teacher, data):
    #     if self.queue_initialized:
    #         print("queue has benn initialized...")
    #         return True
    #
    #     print("## Initializing queue...")
    #     student.eval()
    #     teacher.eval()
    #
    #     pbar = tqdm(range(len(data)))
    #     for idx in pbar:
    #         batch = data[idx]
    #
    #         if USE_CUDA:
    #             token_ids = torch.tensor(batch["token_ids"], dtype=torch.long).cuda()
    #             token_ids_MaskN = torch.tensor(batch["token_ids_MaskN"], dtype=torch.long).cuda()
    #             # token_ids_DisturbN = torch.tensor(batch["token_ids_DisturbN"], dtype=torch.long).cuda()
    #             token_type_ids = torch.tensor(batch["token_type_ids"], dtype=torch.long).cuda()
    #             attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
    #             label = torch.tensor(batch["template"], dtype=torch.long).cuda()
    #
    #         # _, output_ori, _, _, _ = teacher(token_ids, token_type_ids, attention_mask)
    #         # change to
    #         _, _, _, output_ori, _ = teacher(token_ids, token_type_ids, attention_mask)
    #
    #         # _, _, _, output_maskN, _ = student(token_ids_MaskN, token_type_ids, attention_mask) # orig
    #         _, _, _, output_maskN, _ = student(token_ids, token_type_ids, attention_mask)
    #         # _, _, _, output_disturbN, _ = student(token_ids_DisturbN, token_type_ids, attention_mask)
    #
    #         output_ori = output_ori.detach()
    #         output_maskN = output_maskN.detach()
    #         # output_disturbN = output_disturbN.detach()
    #
    #         self.push_queue_teacher(output_ori, label)
    #         self.push_queue_maskN(output_maskN)
    #         # self.push_queue_disturbN(output_disturbN)
    #         self.push_label(label)
    #
    #         if len(self.label) > self.K:
    #             self.pop_queue()
    #
    #     assert len(self.queue_maskN) == len(self.label)
    #     # assert len(self.queue_disturbN) == len(self.label)
    #     self.queue_initialized = True
    #
    #     print("## CRD len: ", len(self.queue_teacher))
    #     print("## moco len: ", len(self.queue_maskN))
    #     print("## label len: ", len(self.label))
    #
    #     return False
    #
    # def contrastive_loss(self, feature_q, feature_k_maskN, template):
    #     # feature: [B x hidden_size]
    #     # template: [B]
    #     N, H = feature_q.shape
    #     print("template:", template.size())
    #     print("queue_maskN:", self.queue_maskN.size())
    #     print("label:", self.label.size())
    #     mask = torch.eq(template.view(N,1).repeat(1, len(self.queue_maskN)), self.label.repeat(N, 1)).float() # [N x K]
    #     print("mask:", mask.size())
    #     print(mask)
    #
    #     logits_qk = torch.bmm(feature_q.view(N, 1, H), feature_k_maskN.view(N, H, 1)).view(N,1)
    #     logits_q_queue = torch.mm(feature_q.view(N, H), self.queue_maskN.transpose(0, 1)) # [N x K]
    #
    #     logits = torch.cat([logits_q_queue, logits_qk], dim=1)/self.T # [N x K+1]
    #     mask_add = torch.ones(N, 1)
    #     if USE_CUDA:
    #         mask_add = mask_add.cuda()
    #     mask_all = torch.cat([mask, mask_add], dim=1) # [N x K+1]
    #     num_positive_row = mask_all.sum(1, keepdim=True)
    #
    #     exp_log = torch.exp(logits) # [N x K+1]
    #     exp_log_sum = torch.sum(exp_log, dim=1, keepdim=True) # [N, 1]
    #     exp_log = torch.log(exp_log/exp_log_sum) # [N x K+1]
    #
    #     log_pi_sum = torch.sum(exp_log*mask_all, dim=1, keepdim=True)/num_positive_row # [N*1]
    #
    #     loss = - torch.mean(log_pi_sum)
    #     return loss
    #
    # def crd_loss(self, feature, template):
    #     # feature: [B x hidden_size]
    #     # template: [B]
    #
    #     N, H = feature.shape
    #     mask_positive = torch.eq(template.view(N,1).repeat(1, len(self.queue_teacher)), self.label_teacher.repeat(N, 1)).float() # [N x train_len]
    #     mask_negative = 1 - mask_positive
    #
    #     num_positive_row = mask_positive.sum(1)
    #     num_negative_row = mask_negative.sum(1)
    #
    #     assert not 0 in num_positive_row
    #     assert not 0 in num_negative_row
    #
    #     exp_dot_st = torch.exp(torch.mm(feature.view(N, H), self.queue_teacher.transpose(0, 1))/self.T) # [N x train_len]
    #     h_ts = exp_dot_st / (exp_dot_st + self.M)
    #
    #     log_h_positive = torch.log(h_ts)
    #     log_h_negative = torch.log(1 - h_ts)
    #
    #     positive_loss = torch.sum(log_h_positive * mask_positive, dim=1) / num_positive_row
    #     negative_loss = torch.sum(log_h_negative * mask_negative, dim=1) / num_negative_row
    #
    #     loss = - torch.mean(positive_loss + negative_loss)
    #     return loss
    #
    # def distill_loss(self, hidden_s, cls_s, hidden_t, cls_t):
    #     B = hidden_s.shape[0]
    #     hidden_s = hidden_s.reshape(B, -1)
    #     hidden_t = hidden_t.reshape(B, -1)
    #
    #     loss_fct = nn.MSELoss()
    #
    #     cls_los = loss_fct(cls_s, cls_t)
    #     hidden_loss = loss_fct(hidden_s, hidden_t)
    #     return cls_los + hidden_loss
