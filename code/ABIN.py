import torch 
import pandas as pd
import pdb
import argparse
import pickle
import os
import numpy as np
import random
import tqdm

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def add_type_relation(is_data):
    ret = []
    for h, t in is_data.values:
        ret.append([h, 'type', t])
    return pd.DataFrame(ret, columns=['h', 'r', 't'])

def get_mapper(root):
    kg_data_all = pd.read_csv(root + 'mid/kg_data_all.csv', index_col=0)
    is_data_all = pd.read_csv(root + 'mid/is_data_all.csv', index_col=0)
    is_data_train = pd.read_csv(root + 'mid/is_data_train.csv', index_col=0)
    ot = pd.read_csv(root + 'mid/ot.csv', index_col=0)

    e_from_kg = set(kg_data_all['h'].unique()) | set(kg_data_all['t'].unique())
    e_from_is = set(is_data_all['h'].unique())
    c_from_is = set(is_data_all['t'].unique())
    c_from_ot = set(ot['h'].unique()) | set(ot['t'].unique())

    e = e_from_kg | e_from_is
    c = c_from_ot | c_from_is
    r = set(kg_data_all['r'].unique())

    e_dict = dict(zip(e, range(len(e))))
    c_dict = dict(zip(c, range(len(e))))
    r_dict = dict(zip(r, range(len(r))))
    print(f'E: 0--{len(e) - 1}, C: 0--{len(c) - 1}, R: {len(r)}')
    return e_dict, c_dict, r_dict, ot, is_data_train

def ppc(data, e_mapper, c_mapper, r_mapper, query_type, answer_type, flag):
    if query_type == '1p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    for answer in data[key]:
                        ret.append([11, 0, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[answer]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    for answer in data[key]:
                        ret.append([12, 0, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], c_mapper[answer]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '2p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([21, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([22, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '3p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([31, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([32, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '2i':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([41, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([42, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '3i':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([51, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([52, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'pi':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([61, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([62, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError
    
    elif query_type == 'ip':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([71, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([72, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError
        
    elif query_type == '2u':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([81, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([82, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'up':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([91, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([92, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = torch.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError
    
    elif query_type == 'ot':
        ret = []
        for c_1, c_2 in data.values.tolist():
            ret.append([0, 0, 0, 0, 0, 0, c_dict[c_1], c_dict[c_2]])
        return torch.tensor(ret)
    
    elif query_type == 'is':
        ret = []
        for e, c in data.values.tolist():
            ret.append([100, 0, 0, 0, 0, 0, e_dict[e], c_dict[c]])
        return torch.tensor(ret)
    
    else:
        raise ValueError

    return ret

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, e_dict, c_dict, data, num_ng, filters):
        super().__init__()
        self.n_entity = len(e_dict)
        self.n_concept = len(c_dict)
        self.data = data
        self.num_ng = num_ng
        self.filters = filters
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data[idx][0] == 0:
            pos = self.data[idx].unsqueeze(dim=0)
            neg_concepts = torch.randint(self.n_concept, (self.num_ng, 1))
            neg = torch.zeros_like(pos).repeat(self.num_ng, 1)
            neg[:2, -2] = neg_concepts[:2, 0]
            neg[:2, -1] = pos[0, -1]
            neg[2:, -1] = neg_concepts[2:, 0]
            neg[2:, -2] = pos[0, -2]
            return torch.cat([pos, neg], dim=0)
        elif self.data[idx][0] == 100:
            pos = self.data[idx].unsqueeze(dim=0)
            neg_concepts = torch.randint(self.n_concept, (self.num_ng//2, 1))
            neg_entities = torch.randint(self.n_entity, (self.num_ng//2, 1))
            neg = torch.zeros_like(pos).repeat(self.num_ng, 1)
            neg[:, 0] = 100
            neg[:2, -2] = neg_entities[:, 0]
            neg[:2, -1] = pos[0, -1]
            neg[2:, -1] = neg_concepts[:, 0]
            neg[2:, -2] = pos[0, -2]
            return torch.cat([pos, neg], dim=0)
        elif self.data[idx][0] == 11:
            flt = self.filters['e']['1p'][(self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 12:
            flt = self.filters['c']['1p'][(self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 21:
            flt = self.filters['e']['2p'][(self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 22:
            flt = self.filters['c']['2p'][(self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 31:
            flt = self.filters['e']['3p'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 32:
            flt = self.filters['c']['3p'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 41:
            flt = self.filters['e']['2i'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 42:
            flt = self.filters['c']['2i'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 51:
            flt = self.filters['e']['3i'][(self.data[idx][1].item(), self.data[idx][2].item(), self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 52:
            flt = self.filters['c']['3i'][(self.data[idx][1].item(), self.data[idx][2].item(), self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        else:
            raise ValueError
        if self.data[idx][0] != 0 and self.data[idx][0] != 100:
            neg_answers = []
            query = self.data[idx][:-1]
            while len(neg_answers) < self.num_ng:
                neg_answer = torch.randint(n, (1, 1))
                if neg_answer.item() in flt:
                    continue
                neg_answers.append(neg_answer)
            neg_answers = torch.cat(neg_answers, dim=0)
            neg = torch.cat([query.expand(self.num_ng, -1), neg_answers], dim=1)
            return torch.cat([self.data[idx].unsqueeze(dim=0), neg], dim=0)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, data, num):
        super().__init__()
        self.n_candidate = num
        self.data = data[:1000]
        self.all_candidate = torch.arange(num).unsqueeze(-1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        return self.data[idx], torch.cat([pos[:-1].expand(self.n_candidate, -1), self.all_candidate], dim=1)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, num):
        super().__init__()
        self.n_candidate = num
        self.data = data[1000:2000]
        self.all_candidate = torch.arange(num).unsqueeze(-1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        return self.data[idx], torch.cat([pos[:-1].expand(self.n_candidate, -1), self.all_candidate], dim=1)


class ABIN(torch.nn.Module):
    def __init__(self, emb_dim, e_dict, c_dict, r_dict):
        super().__init__()
        self.emb_dim = emb_dim
        self.e_dict = e_dict
        self.e_embedding = torch.nn.Embedding(len(e_dict), emb_dim)
        self.c_embedding = torch.nn.Embedding(len(c_dict), emb_dim)
        self.r_embedding = torch.nn.Embedding(len(r_dict), emb_dim)
        self.fc_1 = torch.nn.Linear(emb_dim, emb_dim)
        self.fc_2 = torch.nn.Linear(emb_dim, emb_dim)
        self.cc_fc_1 = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.cc_fc_2 = torch.nn.Linear(emb_dim, 1)

        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_2.weight.data)

    def compute_loss(self, logits):
        pos = logits[:, 0].unsqueeze(dim=-1)
        neg = logits[:, 1:]
        return - torch.nn.functional.logsigmoid(pos - neg).mean()
    
    def js_div(self, fs_1, fs_2):
        fs_1_normalized = torch.nn.functional.normalize(fs_1, p=1, dim=-1)
        fs_2_normalized = torch.nn.functional.normalize(fs_2, p=1, dim=-1)
        M = 0.5 * (fs_1_normalized + fs_2_normalized)
        kl_1 = torch.nn.functional.kl_div(torch.log(fs_1_normalized + 1e-10), M, reduction='none', log_target=False).sum(dim=-1)
        kl_2 = torch.nn.functional.kl_div(torch.log(fs_2_normalized + 1e-10), M, reduction='none', log_target=False).sum(dim=-1)
        return 0.5 * (kl_1 + kl_2)

    def subsumption_loss(self, x):
        x_sub = torch.index_select(x, 0, (x[:, 0, 0] == 0).nonzero().squeeze(-1))
        if len(x_sub):
            c_1 = self.c_embedding(x_sub[:, :, -2])
            c_2 = self.c_embedding(x_sub[:, :, -1])
            cc = torch.cat([c_1, c_2], dim=-1)
            logits = self.cc_fc_2(torch.nn.functional.relu(self.cc_fc_1(cc))).squeeze(dim=-1)
            return self.compute_loss(logits)
        else:
            return 0

    def inclusion_loss(self, x):
        x_inc = torch.index_select(x, 0, (x[:, 0, 0] == 100).nonzero().squeeze(-1))
        if len(x_inc):
            e = self.e_embedding(x_inc[:, :, -2])
            c = self.c_embedding(x_inc[:, :, -1])
            logits = (e * c).sum(dim=-1)
            return self.compute_loss(logits)
        else:
            return 0

    def query_entity_loss(self, x):
        # 1p e
        x_1p = torch.index_select(x, 0, (x[:, 0, 0] == 11).nonzero().squeeze(-1))
        if len(x_1p):
            anchor_1p = self.e_embedding(x_1p[:, :, -3])
            relation_1p = self.r_embedding(x_1p[:, :, -2])
            answer_1p = self.e_embedding(x_1p[:, :, -1])
            logits_1p = - torch.norm(anchor_1p + relation_1p - answer_1p, p=1, dim=-1)
            loss_1p = self.compute_loss(logits_1p)
        else:
            loss_1p = 0
        
        # 2p e
        x_2p = torch.index_select(x, 0, (x[:, 0, 0] == 21).nonzero().squeeze(-1))
        if len(x_2p):
            anchor_2p = self.e_embedding(x_2p[:, :, -4])
            relation_2p_1 = self.r_embedding(x_2p[:, :, -3])
            relation_2p_2 = self.r_embedding(x_2p[:, :, -2])
            answer_2p = self.e_embedding(x_2p[:, :, -1])
            logits_2p = - torch.norm(anchor_2p + relation_2p_1 + relation_2p_2 - answer_2p, p=1, dim=-1)
            loss_2p = self.compute_loss(logits_2p)
        else:
            loss_2p = 0

        # 3p e
        x_3p = torch.index_select(x, 0, (x[:, 0, 0] == 31).nonzero().squeeze(-1))
        if len(x_3p):
            anchor_3p = self.e_embedding(x_3p[:, :, -5])
            relation_3p_1 = self.r_embedding(x_3p[:, :, -4])
            relation_3p_2 = self.r_embedding(x_3p[:, :, -3])
            relation_3p_3 = self.r_embedding(x_3p[:, :, -2])
            answer_3p = self.e_embedding(x_3p[:, :, -1])
            logits_3p = - torch.norm(anchor_3p + relation_3p_1 + relation_3p_2 + relation_3p_3 - answer_3p, p=1, dim=-1)
            loss_3p = self.compute_loss(logits_3p)
        else:
            loss_3p = 0

        # 2i e
        x_2i = torch.index_select(x, 0, (x[:, 0, 0] == 41).nonzero().squeeze(-1))
        if len(x_2i):
            anchor_2i_1 = self.e_embedding(x_2i[:, :, -5])
            relation_2i_1 = self.r_embedding(x_2i[:, :, -4])
            anchor_2i_2 = self.e_embedding(x_2i[:, :, -3])
            relation_2i_2 = self.r_embedding(x_2i[:, :, -2])
            answer_2i = self.e_embedding(x_2i[:, :, -1])

            query_2i_1 = (anchor_2i_1 + relation_2i_1).unsqueeze(dim=2) 
            query_2i_2 = (anchor_2i_2 + relation_2i_2).unsqueeze(dim=2)
            query_2i = torch.cat([query_2i_1, query_2i_2], dim=2)

            mid_2i = torch.nn.functional.relu(self.fc_1(query_2i))
            attention = torch.nn.functional.softmax(self.fc_2(mid_2i), dim=2)
            query_emb_2i = torch.sum(attention * query_2i, dim=2)
            logits_2i = - torch.norm(query_emb_2i - answer_2i, p=1, dim=-1)
            loss_2i = self.compute_loss(logits_2i)
        else:
            loss_2i = 0
        
        # 3i e
        x_3i = torch.index_select(x, 0, (x[:, 0, 0] == 51).nonzero().squeeze(-1))
        if len(x_3i):
            anchor_3i_1 = self.e_embedding(x_3i[:, :, -7])
            relation_3i_1 = self.r_embedding(x_3i[:, :, -6])
            anchor_3i_2 = self.e_embedding(x_3i[:, :, -5])
            relation_3i_2 = self.r_embedding(x_3i[:, :, -4])
            anchor_3i_3 = self.e_embedding(x_3i[:, :, -3])
            relation_3i_3 = self.r_embedding(x_3i[:, :, -2])
            answer_3i = self.e_embedding(x_3i[:, :, -1])

            query_3i_1 = (anchor_3i_1 + relation_3i_1).unsqueeze(dim=2)
            query_3i_2 = (anchor_3i_2 + relation_3i_2).unsqueeze(dim=2)
            query_3i_3 = (anchor_3i_3 + relation_3i_3).unsqueeze(dim=2)

            query_3i = torch.cat([query_3i_1, query_3i_2, query_3i_3], dim=2)
            mid_3i = torch.nn.functional.relu(self.fc_1(query_3i))
            attention = torch.nn.functional.softmax(self.fc_2(mid_3i), dim=2)
            query_emb_3i = torch.sum(attention * query_3i, dim=2)
            logits_3i = - torch.norm(query_emb_3i - answer_3i, p=1, dim=-1)
            loss_3i = self.compute_loss(logits_3i)
        else:
            loss_3i = 0
        
        return [loss_1p, loss_2p, loss_3p, loss_2i, loss_3i]
    
    def query_concept_loss(self, x):
        # 1p c
        x_1p = torch.index_select(x, 0, (x[:, 0, 0] == 12).nonzero().squeeze(-1))
        if len(x_1p):
            anchor_1p = self.e_embedding(x_1p[:, :, -3])
            relation_1p = self.r_embedding(x_1p[:, :, -2])
            answer_1p = self.c_embedding(x_1p[:, :, -1])
            logits_1p = ((anchor_1p + relation_1p) * answer_1p).sum(dim=-1)
            loss_1p = self.compute_loss(logits_1p)
        else:
            loss_1p = 0
        
        # 2p c
        x_2p = torch.index_select(x, 0, (x[:, 0, 0] == 22).nonzero().squeeze(-1))
        if len(x_2p):
            anchor_2p = self.e_embedding(x_2p[:, :, -4])
            relation_2p_1 = self.r_embedding(x_2p[:, :, -3])
            relation_2p_2 = self.r_embedding(x_2p[:, :, -2])
            answer_2p = self.c_embedding(x_2p[:, :, -1])
            logits_2p = ((anchor_2p + relation_2p_1 + relation_2p_2) * answer_2p).sum(dim=-1)
            loss_2p = self.compute_loss(logits_2p)
        else:
            loss_2p = 0

        # 3p c
        x_3p = torch.index_select(x, 0, (x[:, 0, 0] == 32).nonzero().squeeze(-1))
        if len(x_3p):
            anchor_3p = self.e_embedding(x_3p[:, :, -5])
            relation_3p_1 = self.r_embedding(x_3p[:, :, -4])
            relation_3p_2 = self.r_embedding(x_3p[:, :, -3])
            relation_3p_3 = self.r_embedding(x_3p[:, :, -2])
            answer_3p = self.c_embedding(x_3p[:, :, -1])
            logits_3p = ((anchor_3p + relation_3p_1 + relation_3p_2 + relation_3p_3) * answer_3p).sum(dim=-1)
            loss_3p = self.compute_loss(logits_3p)
        else:
            loss_3p = 0

        # 2i c
        x_2i = torch.index_select(x, 0, (x[:, 0, 0] == 42).nonzero().squeeze(-1))
        if len(x_2i):
            anchor_2i_1 = self.e_embedding(x_2i[:, :, -5])
            relation_2i_1 = self.r_embedding(x_2i[:, :, -4])
            anchor_2i_2 = self.e_embedding(x_2i[:, :, -3])
            relation_2i_2 = self.r_embedding(x_2i[:, :, -2])
            answer_2i = self.c_embedding(x_2i[:, :, -1])

            fs_2i_1 = torch.sigmoid(torch.matmul((anchor_2i_1 + relation_2i_1), self.e_embedding.weight.data.t()))
            fs_2i_2 = torch.sigmoid(torch.matmul((anchor_2i_2 + relation_2i_2), self.e_embedding.weight.data.t()))
            fs_2i_q = fs_2i_1 * fs_2i_2
            fs_2i_c = torch.sigmoid(torch.matmul(answer_2i, self.e_embedding.weight.data.t()))
            logits_2i =  - self.js_div(fs_2i_q, fs_2i_c)
            loss_2i = self.compute_loss(logits_2i)
        else:
            loss_2i = 0
        
        # 3i c
        x_3i = torch.index_select(x, 0, (x[:, 0, 0] == 52).nonzero().squeeze(-1))
        if len(x_3i):
            anchor_3i_1 = self.e_embedding(x_3i[:, :, -7])
            relation_3i_1 = self.r_embedding(x_3i[:, :, -6])
            anchor_3i_2 = self.e_embedding(x_3i[:, :, -5])
            relation_3i_2 = self.r_embedding(x_3i[:, :, -4])
            anchor_3i_3 = self.e_embedding(x_3i[:, :, -3])
            relation_3i_3 = self.r_embedding(x_3i[:, :, -2])
            answer_3i = self.c_embedding(x_3i[:, :, -1])

            fs_3i_1 = torch.sigmoid(torch.matmul((anchor_3i_1 + relation_3i_1), self.e_embedding.weight.data.t()))
            fs_3i_2 = torch.sigmoid(torch.matmul((anchor_3i_2 + relation_3i_2), self.e_embedding.weight.data.t()))
            fs_3i_3 = torch.sigmoid(torch.matmul((anchor_3i_3 + relation_3i_3), self.e_embedding.weight.data.t()))
            fs_3i_q = fs_3i_1 * fs_3i_2 * fs_3i_3
            fs_3i_c = torch.sigmoid(torch.matmul(answer_3i, self.e_embedding.weight.data.t()))
            logits_3i =  - self.js_div(fs_3i_q, fs_3i_c)
            loss_3i = self.compute_loss(logits_3i)
        else:
            loss_3i = 0
        
        return [loss_1p, loss_2p, loss_3p, loss_2i, loss_3i]

    def query_entity_logit(self, x, query_type):
        # 1p e
        if query_type == '1p':
            x_1p = torch.index_select(x, 0, (x[:, 0, 0] == 11).nonzero().squeeze(-1))
            anchor_1p = self.e_embedding(x_1p[:, 0, -3])
            relation_1p = self.r_embedding(x_1p[:, 0, -2])
            answer_1p = self.e_embedding(x_1p[:, :, -1])
            logits = - torch.norm(anchor_1p + relation_1p - answer_1p, p=1, dim=-1)
        
        # 2p e
        elif query_type == '2p':
            x_2p = torch.index_select(x, 0, (x[:, 0, 0] == 21).nonzero().squeeze(-1))
            anchor_2p = self.e_embedding(x_2p[:, 0, -4])
            relation_2p_1 = self.r_embedding(x_2p[:, 0, -3])
            relation_2p_2 = self.r_embedding(x_2p[:, 0, -2])
            answer_2p = self.e_embedding(x_2p[:, :, -1])
            logits = - torch.norm(anchor_2p + relation_2p_1 + relation_2p_2 - answer_2p, p=1, dim=-1)

        # 3p e
        elif query_type == '3p':
            x_3p = torch.index_select(x, 0, (x[:, 0, 0] == 31).nonzero().squeeze(-1))
            anchor_3p = self.e_embedding(x_3p[:, 0, -5])
            relation_3p_1 = self.r_embedding(x_3p[:, 0, -4])
            relation_3p_2 = self.r_embedding(x_3p[:, 0, -3])
            relation_3p_3 = self.r_embedding(x_3p[:, 0, -2])
            answer_3p = self.e_embedding(x_3p[:, :, -1])
            logits = - torch.norm(anchor_3p + relation_3p_1 + relation_3p_2 + relation_3p_3 - answer_3p, p=1, dim=-1)

        # 2i e
        elif query_type == '2i':
            x_2i = torch.index_select(x, 0, (x[:, 0, 0] == 41).nonzero().squeeze(-1))
            anchor_2i_1 = self.e_embedding(x_2i[:, 0, -5])
            relation_2i_1 = self.r_embedding(x_2i[:, 0, -4])
            anchor_2i_2 = self.e_embedding(x_2i[:, 0, -3])
            relation_2i_2 = self.r_embedding(x_2i[:, 0, -2])
            answer_2i = self.e_embedding(x_2i[:, :, -1])

            query_2i_1 = (anchor_2i_1 + relation_2i_1).unsqueeze(dim=1)
            query_2i_2 = (anchor_2i_2 + relation_2i_2).unsqueeze(dim=1)
            query_2i = torch.cat([query_2i_1, query_2i_2], dim=1)

            mid_2i = torch.nn.functional.relu(self.fc_1(query_2i))
            attention = torch.nn.functional.softmax(self.fc_2(mid_2i), dim=1)
            query_emb_2i = torch.sum(attention * query_2i, dim=1)
            logits = - torch.norm(query_emb_2i.unsqueeze(dim=1) - answer_2i, p=1, dim=-1)
        
        # 3i e
        elif query_type == '3i':
            x_3i = torch.index_select(x, 0, (x[:, 0, 0] == 51).nonzero().squeeze(-1))
            anchor_3i_1 = self.e_embedding(x_3i[:, 0, -7])
            relation_3i_1 = self.r_embedding(x_3i[:, 0, -6])
            anchor_3i_2 = self.e_embedding(x_3i[:, 0, -5])
            relation_3i_2 = self.r_embedding(x_3i[:, 0, -4])
            anchor_3i_3 = self.e_embedding(x_3i[:, 0, -3])
            relation_3i_3 = self.r_embedding(x_3i[:, 0, -2])
            answer_3i = self.e_embedding(x_3i[:, :, -1])

            query_3i_1 = (anchor_3i_1 + relation_3i_1).unsqueeze(dim=1)
            query_3i_2 = (anchor_3i_2 + relation_3i_2).unsqueeze(dim=1)
            query_3i_3 = (anchor_3i_3 + relation_3i_3).unsqueeze(dim=1)

            query_3i = torch.cat([query_3i_1, query_3i_2, query_3i_3], dim=1)
            mid_3i = torch.nn.functional.relu(self.fc_1(query_3i))
            attention = torch.nn.functional.softmax(self.fc_2(mid_3i), dim=1)
            query_emb_3i = torch.sum(attention * query_3i, dim=1)
            logits = - torch.norm(query_emb_3i.unsqueeze(dim=1) - answer_3i, p=1, dim=-1)
        
        # pi e
        elif query_type == 'pi':
            x_pi = torch.index_select(x, 0, (x[:, 0, 0] == 61).nonzero().squeeze(-1))
            anchor_pi_1 = self.e_embedding(x_pi[:, 0, -6])
            relation_pi_11 = self.r_embedding(x_pi[:, 0, -5])
            relation_pi_12 = self.r_embedding(x_pi[:, 0, -4])
            anchor_pi_2 = self.e_embedding(x_pi[:, 0, -3])
            relation_pi_2 = self.r_embedding(x_pi[:, 0, -2])
            answer_pi = self.e_embedding(x_pi[:, :, -1])

            query_pi_1 = (anchor_pi_1 + relation_pi_11 + relation_pi_11).unsqueeze(dim=1)
            query_pi_2 = (anchor_pi_2 + relation_pi_2).unsqueeze(dim=1)
            query_pi = torch.cat([query_pi_1, query_pi_2], dim=1)

            mid_pi = torch.nn.functional.relu(self.fc_1(query_pi))
            attention = torch.nn.functional.softmax(self.fc_2(mid_pi), dim=1)
            query_emb_pi = torch.sum(attention * query_pi, dim=1)
            logits = - torch.norm(query_emb_pi.unsqueeze(dim=1) - answer_pi, p=1, dim=-1)
        
        # ip e
        elif query_type == 'ip':
            x_ip = torch.index_select(x, 0, (x[:, 0, 0] == 71).nonzero().squeeze(-1))
            anchor_ip_1 = self.e_embedding(x_ip[:, 0, -6])
            relation_ip_1 = self.r_embedding(x_ip[:, 0, -5])
            anchor_ip_2 = self.e_embedding(x_ip[:, 0, -4])
            relation_ip_2 = self.r_embedding(x_ip[:, 0, -3])
            relation_ip_3 = self.r_embedding(x_ip[:, 0, -2])
            answer_ip = self.e_embedding(x_ip[:, :, -1])

            query_ip_1 = (anchor_ip_1 + relation_ip_1 + relation_ip_3).unsqueeze(dim=1)
            query_ip_2 = (anchor_ip_2 + relation_ip_2 + relation_ip_3).unsqueeze(dim=1)
            query_ip = torch.cat([query_ip_1, query_ip_2], dim=1)

            mid_ip = torch.nn.functional.relu(self.fc_1(query_ip))
            attention = torch.nn.functional.softmax(self.fc_2(mid_ip), dim=1)
            query_emb_ip = torch.sum(attention * query_ip, dim=1)
            logits = - torch.norm(query_emb_ip.unsqueeze(dim=1) - answer_ip, p=1, dim=-1)
        
        # 2u e
        elif query_type == '2u':
            x_2u = torch.index_select(x, 0, (x[:, 0, 0] == 81).nonzero().squeeze(-1))
            anchor_2u_1 = self.e_embedding(x_2u[:, 0, -5])
            relation_2u_1 = self.r_embedding(x_2u[:, 0, -4])
            anchor_2u_2 = self.e_embedding(x_2u[:, 0, -3])
            relation_2u_2 = self.r_embedding(x_2u[:, 0, -2])
            answer_2u = self.e_embedding(x_2u[:, :, -1])

            logits_2u_1 = - torch.norm(anchor_2u_1 + relation_2u_1 - answer_2u, p=1, dim=-1).unsqueeze(dim=-1)
            logits_2u_2 = - torch.norm(anchor_2u_2 + relation_2u_2 - answer_2u, p=1, dim=-1).unsqueeze(dim=-1)
            logits = torch.cat([logits_2u_1, logits_2u_2], dim=-1).max(dim=-1)[0]

        # up e
        elif query_type == 'up':
            x_up = torch.index_select(x, 0, (x[:, 0, 0] == 91).nonzero().squeeze(-1))
            anchor_up_1 = self.e_embedding(x_up[:, 0, -6])
            relation_up_1 = self.r_embedding(x_up[:, 0, -5])
            anchor_up_2 = self.e_embedding(x_up[:, 0, -4])
            relation_up_2 = self.r_embedding(x_up[:, 0, -3])
            relation_up_3 = self.r_embedding(x_up[:, 0, -2])
            answer_up = self.e_embedding(x_up[:, :, -1])

            logits_up_1 = - torch.norm(anchor_up_1 + relation_up_1 + relation_up_3 - answer_up, p=1, dim=-1).unsqueeze(dim=-1)
            logits_up_2 = - torch.norm(anchor_up_2 + relation_up_2 + relation_up_3 - answer_up, p=1, dim=-1).unsqueeze(dim=-1)
            logits = torch.cat([logits_up_1, logits_up_2], dim=-1).max(dim=-1)[0]
        
        return logits

    def query_concept_logit(self, x, query_type):
        # 1p c
        if query_type == '1p':
            x_1p = torch.index_select(x, 0, (x[:, 0, 0] == 12).nonzero().squeeze(-1))
            anchor_1p = self.e_embedding(x_1p[:, 0, -3])
            relation_1p = self.r_embedding(x_1p[:, 0, -2])
            answer_1p = self.c_embedding(x_1p[:, :, -1])
            logits = ((anchor_1p + relation_1p) * answer_1p).sum(dim=-1)
        
        # 2p c
        elif query_type == '2p':
            x_2p = torch.index_select(x, 0, (x[:, 0, 0] == 22).nonzero().squeeze(-1))
            anchor_2p = self.e_embedding(x_2p[:, 0, -4])
            relation_2p_1 = self.r_embedding(x_2p[:, 0, -3])
            relation_2p_2 = self.r_embedding(x_2p[:, 0, -2])
            answer_2p = self.c_embedding(x_2p[:, :, -1])
            logits = ((anchor_2p + relation_2p_1 + relation_2p_2) * answer_2p).sum(dim=-1)

        # 3p c
        elif query_type == '3p':
            x_3p = torch.index_select(x, 0, (x[:, 0, 0] == 32).nonzero().squeeze(-1))
            anchor_3p = self.e_embedding(x_3p[:, 0, -5])
            relation_3p_1 = self.r_embedding(x_3p[:, 0, -4])
            relation_3p_2 = self.r_embedding(x_3p[:, 0, -3])
            relation_3p_3 = self.r_embedding(x_3p[:, 0, -2])
            answer_3p = self.c_embedding(x_3p[:, :, -1])
            logits = ((anchor_3p + relation_3p_1 + relation_3p_2 + relation_3p_3) * answer_3p).sum(dim=-1)

        # 2i c
        elif query_type == '2i':
            x_2i = torch.index_select(x, 0, (x[:, 0, 0] == 42).nonzero().squeeze(-1))
            anchor_2i_1 = self.e_embedding(x_2i[:, 0, -5])
            relation_2i_1 = self.r_embedding(x_2i[:, 0, -4])
            anchor_2i_2 = self.e_embedding(x_2i[:, 0, -3])
            relation_2i_2 = self.r_embedding(x_2i[:, 0, -2])
            answer_2i = self.c_embedding(x_2i[:, :, -1])

            fs_2i_1 = torch.sigmoid(torch.matmul((anchor_2i_1 + relation_2i_1), self.e_embedding.weight.data.t()))
            fs_2i_2 = torch.sigmoid(torch.matmul((anchor_2i_2 + relation_2i_2), self.e_embedding.weight.data.t()))
            fs_2i_q = fs_2i_1 * fs_2i_2
            fs_2i_c = torch.sigmoid(torch.matmul(answer_2i, self.e_embedding.weight.data.t()))
            logits = - self.js_div(fs_2i_q, fs_2i_c)
        
        # 3i c
        elif query_type == '3i':
            x_3i = torch.index_select(x, 0, (x[:, 0, 0] == 52).nonzero().squeeze(-1))
            anchor_3i_1 = self.e_embedding(x_3i[:, 0, -7])
            relation_3i_1 = self.r_embedding(x_3i[:, 0, -6])
            anchor_3i_2 = self.e_embedding(x_3i[:, 0, -5])
            relation_3i_2 = self.r_embedding(x_3i[:, 0, -4])
            anchor_3i_3 = self.e_embedding(x_3i[:, 0, -3])
            relation_3i_3 = self.r_embedding(x_3i[:, 0, -2])
            answer_3i = self.c_embedding(x_3i[:, :, -1])

            fs_3i_1 = torch.sigmoid(torch.matmul((anchor_3i_1 + relation_3i_1), self.e_embedding.weight.data.t()))
            fs_3i_2 = torch.sigmoid(torch.matmul((anchor_3i_2 + relation_3i_2), self.e_embedding.weight.data.t()))
            fs_3i_3 = torch.sigmoid(torch.matmul((anchor_3i_3 + relation_3i_3), self.e_embedding.weight.data.t()))
            fs_3i_q = fs_3i_1 * fs_3i_2 * fs_3i_3
            fs_3i_c = torch.sigmoid(torch.matmul(answer_3i, self.e_embedding.weight.data.t()))
            logits = - self.js_div(fs_3i_q, fs_3i_c)

        # pi c
        elif query_type == 'pi':
            x_pi = torch.index_select(x, 0, (x[:, 0, 0] == 62).nonzero().squeeze(-1))
            anchor_pi_1 = self.e_embedding(x_pi[:, 0, -6])
            relation_pi_11 = self.r_embedding(x_pi[:, 0, -5])
            relation_pi_12 = self.r_embedding(x_pi[:, 0, -4])
            anchor_pi_2 = self.e_embedding(x_pi[:, 0, -3])
            relation_pi_2 = self.r_embedding(x_pi[:, 0, -2])
            answer_pi = self.c_embedding(x_pi[:, :, -1])

            fs_pi_1 = torch.sigmoid(torch.matmul((anchor_pi_1 + relation_pi_11 + relation_pi_12), self.e_embedding.weight.data.t()))
            fs_pi_2 = torch.sigmoid(torch.matmul((anchor_pi_2 + relation_pi_2), self.e_embedding.weight.data.t()))
            fs_pi_q = fs_pi_1 * fs_pi_2
            fs_pi_c = torch.sigmoid(torch.matmul(answer_pi, self.e_embedding.weight.data.t()))
            logits = - self.js_div(fs_pi_q, fs_pi_c)

        # ip c
        elif query_type == 'ip':
            x_ip = torch.index_select(x, 0, (x[:, 0, 0] == 72).nonzero().squeeze(-1))
            anchor_ip_1 = self.e_embedding(x_ip[:, 0, -6])
            relation_ip_1 = self.r_embedding(x_ip[:, 0, -5])
            anchor_ip_2 = self.e_embedding(x_ip[:, 0, -4])
            relation_ip_2 = self.r_embedding(x_ip[:, 0, -3])
            relation_ip_3 = self.r_embedding(x_ip[:, 0, -2])
            answer_ip = self.c_embedding(x_ip[:, :, -1])

            fs_ip_1 = torch.sigmoid(torch.matmul((anchor_ip_1 + relation_ip_1 + relation_ip_3), self.e_embedding.weight.data.t()))
            fs_ip_2 = torch.sigmoid(torch.matmul((anchor_ip_2 + relation_ip_2 + relation_ip_3), self.e_embedding.weight.data.t()))
            fs_ip_q = fs_ip_1 * fs_ip_2
            fs_ip_c = torch.sigmoid(torch.matmul(answer_ip, self.e_embedding.weight.data.t()))
            logits = - self.js_div(fs_ip_q, fs_ip_c)
        
        # 2u c
        elif query_type == '2u':
            x_2u = torch.index_select(x, 0, (x[:, 0, 0] == 82).nonzero().squeeze(-1))
            anchor_2u_1 = self.e_embedding(x_2u[:, 0, -5])
            relation_2u_1 = self.r_embedding(x_2u[:, 0, -4])
            anchor_2u_2 = self.e_embedding(x_2u[:, 0, -3])
            relation_2u_2 = self.r_embedding(x_2u[:, 0, -2])
            answer_2u = self.c_embedding(x_2u[:, :, -1])

            logits_2u_1 = ((anchor_2u_1 + relation_2u_1) * answer_2u).sum(dim=-1).unsqueeze(dim=-1)
            logits_2u_2 = ((anchor_2u_2 + relation_2u_2) * answer_2u).sum(dim=-1).unsqueeze(dim=-1)
            logits = torch.cat([logits_2u_1, logits_2u_2], dim=-1).max(dim=-1)[0]

        # up c
        elif query_type == 'up':
            x_up = torch.index_select(x, 0, (x[:, 0, 0] == 92).nonzero().squeeze(-1))
            anchor_up_1 = self.e_embedding(x_up[:, 0, -6])
            relation_up_1 = self.r_embedding(x_up[:, 0, -5])
            anchor_up_2 = self.e_embedding(x_up[:, 0, -4])
            relation_up_2 = self.r_embedding(x_up[:, 0, -3])
            relation_up_3 = self.r_embedding(x_up[:, 0, -2])
            answer_up = self.c_embedding(x_up[:, :, -1])

            logits_up_1 = ((anchor_up_1 + relation_up_1 + relation_up_3) * answer_up).sum(dim=-1).unsqueeze(dim=-1)
            logits_up_2 = ((anchor_up_2 + relation_up_2) * answer_up).sum(dim=-1).unsqueeze(dim=-1)
            logits = torch.cat([logits_up_1, logits_up_2], dim=-1).max(dim=-1)[0]

        return logits

    def forward(self, x):
        qe_losses = self.query_entity_loss(x)
        qc_losses = self.query_concept_loss(x)
        cc_loss = self.subsumption_loss(x)
        ec_loss = self.inclusion_loss(x)
        return qe_losses, qc_losses, cc_loss, ec_loss
    
    def predict(self, x, query_type, answer_type):
        if answer_type == 'e':
            logits = self.query_entity_logit(x, query_type)
        elif answer_type == 'c':
            logits = self.query_concept_logit(x, query_type)
        return logits


def evaluate_e(model, loader, filters_e, device, query_type):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if cfg.verbose == 1:
        loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='e')
                filter_e = filters_e[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='e')
                filter_e = filters_e[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='e')
                filter_e = filters_e[(pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            else:
                raise ValueError
            ranks = torch.argsort(logits.squeeze(dim=0), descending=True)
            rank = (ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = ranks[:rank - 1]
            for t in filter_e:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            r.append(rank)
            rr.append(1/rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    r = int(sum(r)/len(r))
    rr = round(sum(rr)/len(rr), 3)
    h1 = round(sum(h1)/len(h1), 3)
    h3 = round(sum(h3)/len(h3), 3)
    h10 = round(sum(h10)/len(h10), 3)
    print(f'#Entity#{query_type}# MRR: {rr}, H1: {h1}, H3: {h3}')
    return r, rr, h1, h3, h10

def evaluate_c(model, loader, filters_c, device, query_type):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if cfg.verbose == 1:
        loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='c')
                filter_c = filters_c[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='c')
                filter_c = filters_c[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='c')
                filter_c = filters_c[(pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            else:
                raise ValueError
            ranks = torch.argsort(logits.squeeze(dim=0), descending=True)
            rank = (ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = ranks[:rank - 1]
            for t in filter_c:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            r.append(rank)
            rr.append(1/rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    r = int(sum(r)/len(r))
    rr = round(sum(rr)/len(rr), 3)
    h1 = round(sum(h1)/len(h1), 3)
    h3 = round(sum(h3)/len(h3), 3)
    h10 = round(sum(h10)/len(h10), 3)
    print(f'#Concept#{query_type}# MRR: {rr}, H1: {h1}, H3: {h3}')
    return r, rr, h1, h3, h10

def load_train_and_test_data(root, num_workers, e_dict, c_dict, r_dict, query_type):
    data = load_obj(root + 'input/' + query_type + '.pkl')
    train_e = ppc(data['train']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    train_c = ppc(data['train']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    train_filter_e = ppc(data['train_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    train_filter_c = ppc(data['train_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    test_e = ppc(data['test']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    test_c = ppc(data['test']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    test_filter_e = ppc(data['test_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    test_filter_c = ppc(data['test_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    valid_dataset_e = ValidDataset(test_e, len(e_dict))
    valid_dataloader_e = torch.utils.data.DataLoader(dataset=valid_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    valid_dataset_c = ValidDataset(test_c, len(c_dict))
    valid_dataloader_c = torch.utils.data.DataLoader(dataset=valid_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_e = TestDataset(test_e, len(e_dict))
    test_dataloader_e = torch.utils.data.DataLoader(dataset=test_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_c = TestDataset(test_c, len(c_dict))
    test_dataloader_c = torch.utils.data.DataLoader(dataset=test_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    return train_e, train_c, train_filter_e, train_filter_c, test_e, test_c, test_filter_e, test_filter_c, valid_dataloader_e, valid_dataloader_c, test_dataloader_e, test_dataloader_c

def load_test_data(root, num_workers, e_dict, c_dict, r_dict, query_type):
    data = load_obj(root + 'input/' + query_type + '.pkl')
    test_e = ppc(data['test']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    test_c = ppc(data['test']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    test_filter_e = ppc(data['test_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    test_filter_c = ppc(data['test_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    valid_dataset_e = ValidDataset(test_e, len(e_dict))
    valid_dataloader_e = torch.utils.data.DataLoader(dataset=valid_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    valid_dataset_c = ValidDataset(test_c, len(c_dict))
    valid_dataloader_c = torch.utils.data.DataLoader(dataset=valid_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_e = TestDataset(test_e, len(e_dict))
    test_dataloader_e = torch.utils.data.DataLoader(dataset=test_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_c = TestDataset(test_c, len(c_dict))
    test_dataloader_c = torch.utils.data.DataLoader(dataset=test_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    return test_e, test_c, test_filter_e, test_filter_c, valid_dataloader_e, valid_dataloader_c, test_dataloader_e, test_dataloader_c

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='YAGO4', type=str)
    parser.add_argument('--cc', default=1, type=int)
    parser.add_argument('--ec', default=1, type=int)
    parser.add_argument('--root', default='../../data/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=3000, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--valid_interval', default=50, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--tolerance', default=3, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)

    save_root = '../tmp/'
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    e_dict, c_dict, r_dict, ot, is_data_train = get_mapper(cfg.root + cfg.dataset + '/')

    train_ot = ppc(ot, e_dict, c_dict, r_dict, query_type='ot', answer_type=None, flag=None)
    train_is = ppc(is_data_train, e_dict, c_dict, r_dict, query_type='is', answer_type=None, flag=None)

    train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dataloader_1p_e, valid_dataloader_1p_c, test_dataloader_1p_e, test_dataloader_1p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
    train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dataloader_2p_e, valid_dataloader_2p_c, test_dataloader_2p_e, test_dataloader_2p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
    train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dataloader_3p_e, valid_dataloader_3p_c, test_dataloader_3p_e, test_dataloader_3p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
    train_e_2i, train_c_2i, train_filter_e_2i, train_filter_c_2i, test_e_2i, test_c_2i, test_filter_e_2i, test_filter_c_2i, valid_dataloader_2i_e, valid_dataloader_2i_c, test_dataloader_2i_e, test_dataloader_2i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2i')
    train_e_3i, train_c_3i, train_filter_e_3i, train_filter_c_3i, test_e_3i, test_c_3i, test_filter_e_3i, test_filter_c_3i, valid_dataloader_3i_e, valid_dataloader_3i_c, test_dataloader_3i_e, test_dataloader_3i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3i')
    test_e_pi, test_c_pi, test_filter_e_pi, test_filter_c_pi, valid_dataloader_pi_e, valid_dataloader_pi_c, test_dataloader_pi_e, test_dataloader_pi_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='pi')
    test_e_ip, test_c_ip, test_filter_e_ip, test_filter_c_ip, valid_dataloader_ip_e, valid_dataloader_ip_c, test_dataloader_ip_e, test_dataloader_ip_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='ip')
    test_e_2u, test_c_2u, test_filter_e_2u, test_filter_c_2u, valid_dataloader_2u_e, valid_dataloader_2u_c, test_dataloader_2u_e, test_dataloader_2u_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2u')
    test_e_up, test_c_up, test_filter_e_up, test_filter_c_up, valid_dataloader_up_e, valid_dataloader_up_c, test_dataloader_up_e, test_dataloader_up_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='up')

    train_data = torch.cat([train_ot, train_is, train_e_1p, train_c_1p, train_e_2p, train_c_2p, train_e_3p, train_c_3p, train_e_2i, train_c_2i, train_e_3i, train_c_3i], dim=0)
    train_filters_e = {'1p': train_filter_e_1p, '2p': train_filter_e_2p, '3p': train_filter_e_3p, '2i': train_filter_e_2i, '3i': train_filter_e_3i}
    train_filters_c = {'1p': train_filter_c_1p, '2p': train_filter_c_2p, '3p': train_filter_c_3p, '2i': train_filter_c_2i, '3i': train_filter_c_3i}
    train_dataset = TrainDataset(e_dict, c_dict, train_data, num_ng=cfg.num_ng, filters={'e': train_filters_e, 'c': train_filters_c})
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.bs, num_workers=cfg.num_workers, shuffle=True, drop_last=True)
    
    model = ABIN(cfg.emb_dim, e_dict, c_dict, r_dict)
    model = model.to(device)
    tolerance = cfg.tolerance
    max_rr = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        if cfg.verbose == 1:
            train_dataloader = tqdm.tqdm(train_dataloader)
        for batch in train_dataloader:
            batch = batch.to(device)
            qe_losses, qc_losses, cc_loss, ec_loss = model(batch)
            qe_loss = sum(qe_losses) / len(qe_losses)
            qc_loss = sum(qc_losses) / len(qc_losses)
            if cfg.cc == 0 and cfg.ec == 0:
                loss = (qe_loss + qc_loss) / 2
            elif cfg.cc == 1 and cfg.ec == 0:
                loss = (qe_loss + qc_loss + cc_loss) / 3
            elif cfg.cc == 0 and cfg.ec == 1:
                loss = (qe_loss + qc_loss + ec_loss) / 3
            else:
                loss = (qe_loss + qc_loss + cc_loss + ec_loss) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss)/len(avg_loss), 4)}')
        if (epoch + 1) % cfg.valid_interval == 0:
            model.eval()
            print('Validating Entity Answering:')
            _, rr_1p_e, h1_1p_e, h3_1p_e, _ = evaluate_e(model, valid_dataloader_1p_e, test_filter_e_1p, device, query_type='1p')
            _, rr_2p_e, h1_2p_e, h3_2p_e, _ = evaluate_e(model, valid_dataloader_2p_e, test_filter_e_2p, device, query_type='2p')
            _, rr_3p_e, h1_3p_e, h3_3p_e, _ = evaluate_e(model, valid_dataloader_3p_e, test_filter_e_3p, device, query_type='3p')
            _, rr_2i_e, h1_2i_e, h3_2i_e, _ = evaluate_e(model, valid_dataloader_2i_e, test_filter_e_2i, device, query_type='2i')
            _, rr_3i_e, h1_3i_e, h3_3i_e, _ = evaluate_e(model, valid_dataloader_3i_e, test_filter_e_3i, device, query_type='3i')
            _, rr_pi_e, h1_pi_e, h3_pi_e, _ = evaluate_e(model, valid_dataloader_pi_e, test_filter_e_pi, device, query_type='pi')
            _, rr_ip_e, h1_ip_e, h3_ip_e, _ = evaluate_e(model, valid_dataloader_ip_e, test_filter_e_ip, device, query_type='ip')
            _, rr_2u_e, h1_2u_e, h3_2u_e, _ = evaluate_e(model, valid_dataloader_2u_e, test_filter_e_2u, device, query_type='2u')
            _, rr_up_e, h1_up_e, h3_up_e, _ = evaluate_e(model, valid_dataloader_up_e, test_filter_e_up, device, query_type='up')
            mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e, rr_ip_e, rr_2u_e, rr_up_e]) / 9, 3)
            mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e, h1_ip_e, h1_2u_e, h1_up_e]) / 9, 3)
            mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e, h3_ip_e, h3_2u_e, h3_up_e]) / 9, 3)
            print(f'Entity Answering Mean: \n MRR: {mrr_e}, H1: {mh1_e}, H3: {mh3_e}')
            print('Validating Concept Answering:')
            _, rr_1p_c, h1_1p_c, h3_1p_c, _ = evaluate_c(model, valid_dataloader_1p_c, test_filter_c_1p, device, query_type='1p')
            _, rr_2p_c, h1_2p_c, h3_2p_c, _ = evaluate_c(model, valid_dataloader_2p_c, test_filter_c_2p, device, query_type='2p')
            _, rr_3p_c, h1_3p_c, h3_3p_c, _ = evaluate_c(model, valid_dataloader_3p_c, test_filter_c_3p, device, query_type='3p')
            _, rr_2i_c, h1_2i_c, h3_2i_c, _ = evaluate_c(model, valid_dataloader_2i_c, test_filter_c_2i, device, query_type='2i')
            _, rr_3i_c, h1_3i_c, h3_3i_c, _ = evaluate_c(model, valid_dataloader_3i_c, test_filter_c_3i, device, query_type='3i')
            _, rr_pi_c, h1_pi_c, h3_pi_c, _ = evaluate_c(model, valid_dataloader_pi_c, test_filter_c_pi, device, query_type='pi')
            _, rr_ip_c, h1_ip_c, h3_ip_c, _ = evaluate_c(model, valid_dataloader_ip_c, test_filter_c_ip, device, query_type='ip')
            _, rr_2u_c, h1_2u_c, h3_2u_c, _ = evaluate_c(model, valid_dataloader_2u_c, test_filter_c_2u, device, query_type='2u')
            _, rr_up_c, h1_up_c, h3_up_c, _ = evaluate_c(model, valid_dataloader_up_c, test_filter_c_up, device, query_type='up')
            mrr_c = round(sum([rr_1p_c, rr_2p_c, rr_3p_c, rr_2i_c, rr_3i_c, rr_pi_c, rr_ip_c, rr_2u_c, rr_up_c]) / 9, 3)
            mh1_c = round(sum([h1_1p_c, h1_2p_c, h1_3p_c, h1_2i_c, h1_3i_c, h1_pi_c, h1_ip_c, h1_2u_c, h1_up_c]) / 9, 3)
            mh3_c = round(sum([h3_1p_c, h3_2p_c, h3_3p_c, h3_2i_c, h3_3i_c, h3_pi_c, h3_ip_c, h3_2u_c, h3_up_c]) / 9, 3)
            print(f'Concept Answering Mean: \n MRR: {mrr_c}, H1: {mh1_c}, H3: {mh3_c}')

            if mrr_e >= max_rr:
                max_rr = mrr_e
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            
            torch.save(model.state_dict(), save_root + '_cc' + str(cfg.cc) + '_ec' + str(cfg.ec) + '_' + (str(epoch + 1)))

        if tolerance == 0:
            print(f'Best performance at epoch {epoch - cfg.tolerance * cfg.valid_interval + 1}')
            model.load_state_dict(torch.load(save_root + '_cc' + str(cfg.cc) + '_ec' + str(cfg.ec) + '_' str(epoch - cfg.tolerance * cfg.valid_interval + 1)))
            model.eval()
            print('Testing Entity Answering:')
            _, rr_1p_e, h1_1p_e, h3_1p_e, _ = evaluate_e(model, test_dataloader_1p_e, test_filter_e_1p, device, query_type='1p')
            _, rr_2p_e, h1_2p_e, h3_2p_e, _ = evaluate_e(model, test_dataloader_2p_e, test_filter_e_2p, device, query_type='2p')
            _, rr_3p_e, h1_3p_e, h3_3p_e, _ = evaluate_e(model, test_dataloader_3p_e, test_filter_e_3p, device, query_type='3p')
            _, rr_2i_e, h1_2i_e, h3_2i_e, _ = evaluate_e(model, test_dataloader_2i_e, test_filter_e_2i, device, query_type='2i')
            _, rr_3i_e, h1_3i_e, h3_3i_e, _ = evaluate_e(model, test_dataloader_3i_e, test_filter_e_3i, device, query_type='3i')
            _, rr_pi_e, h1_pi_e, h3_pi_e, _ = evaluate_e(model, test_dataloader_pi_e, test_filter_e_pi, device, query_type='pi')
            _, rr_ip_e, h1_ip_e, h3_ip_e, _ = evaluate_e(model, test_dataloader_ip_e, test_filter_e_ip, device, query_type='ip')
            _, rr_2u_e, h1_2u_e, h3_2u_e, _ = evaluate_e(model, test_dataloader_2u_e, test_filter_e_2u, device, query_type='2u')
            _, rr_up_e, h1_up_e, h3_up_e, _ = evaluate_e(model, test_dataloader_up_e, test_filter_e_up, device, query_type='up')
            mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e, rr_ip_e, rr_2u_e, rr_up_e]) / 9, 3)
            mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e, h1_ip_e, h1_2u_e, h1_up_e]) / 9, 3)
            mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e, h3_ip_e, h3_2u_e, h3_up_e]) / 9, 3)
            print(f'Entity Answering Mean: \n MRR: {mrr_e}, H1: {mh1_e}, H3: {mh3_e}')
            print('Testing Concept Answering:')
            _, rr_1p_c, h1_1p_c, h3_1p_c, _ = evaluate_c(model, test_dataloader_1p_c, test_filter_c_1p, device, query_type='1p')
            _, rr_2p_c, h1_2p_c, h3_2p_c, _ = evaluate_c(model, test_dataloader_2p_c, test_filter_c_2p, device, query_type='2p')
            _, rr_3p_c, h1_3p_c, h3_3p_c, _ = evaluate_c(model, test_dataloader_3p_c, test_filter_c_3p, device, query_type='3p')
            _, rr_2i_c, h1_2i_c, h3_2i_c, _ = evaluate_c(model, test_dataloader_2i_c, test_filter_c_2i, device, query_type='2i')
            _, rr_3i_c, h1_3i_c, h3_3i_c, _ = evaluate_c(model, test_dataloader_3i_c, test_filter_c_3i, device, query_type='3i')
            _, rr_pi_c, h1_pi_c, h3_pi_c, _ = evaluate_c(model, test_dataloader_pi_c, test_filter_c_pi, device, query_type='pi')
            _, rr_ip_c, h1_ip_c, h3_ip_c, _ = evaluate_c(model, test_dataloader_ip_c, test_filter_c_ip, device, query_type='ip')
            _, rr_2u_c, h1_2u_c, h3_2u_c, _ = evaluate_c(model, test_dataloader_2u_c, test_filter_c_2u, device, query_type='2u')
            _, rr_up_c, h1_up_c, h3_up_c, _ = evaluate_c(model, test_dataloader_up_c, test_filter_c_up, device, query_type='up')
            mrr_c = round(sum([rr_1p_c, rr_2p_c, rr_3p_c, rr_2i_c, rr_3i_c, rr_pi_c, rr_ip_c, rr_2u_c, rr_up_c]) / 9, 3)
            mh1_c = round(sum([h1_1p_c, h1_2p_c, h1_3p_c, h1_2i_c, h1_3i_c, h1_pi_c, h1_ip_c, h1_2u_c, h1_up_c]) / 9, 3)
            mh3_c = round(sum([h3_1p_c, h3_2p_c, h3_3p_c, h3_2i_c, h3_3i_c, h3_pi_c, h3_ip_c, h3_2u_c, h3_up_c]) / 9, 3)
            print(f'Concept Answering Mean: \n MRR: {mrr_c}, H1: {mh1_c}, H3: {mh3_c}')
            break
