#coding: utf-8
import os
import pandas as pd

from collections import defaultdict

import torch
from torch import nn
from d2l import torch as d2l

type_dict = {'/organization/company': ['/organization/company/news',
              '/organization/company/broadcast'],
             '/other': ['/other/body_part',
              '/other/currency',
              '/other/product/mobile_phone',
              '/other/language/programming_language',
              '/other/health',
              '/other/art/writing',
              '/other/event',
              '/other/art/broadcast',
              '/other/art',
              '/other/product/computer',
              '/other/product',
              '/other/scientific',
              '/other/product/software',
              '/other/health/malady',
              '/other/event/holiday',
              '/other/event/natural_disaster',
              '/other/religion',
              '/other/food',
              '/other/event/election',
              '/other/art/music',
              '/other/event/violent_conflict',
              '/other/event/accident',
              '/other/heritage',
              '/other/sports_and_leisure',
              '/other/product/weapon',
              '/other/language',
              '/other/living_thing',
              '/other/internet',
              '/other/supernatural',
              '/other/product/car',
              '/other/award',
              '/other/art/stage',
              '/other/event/sports_event',
              '/other/legal',
              '/other/living_thing/animal',
              '/other/art/film',
              '/other/event/protest',
              '/other/health/treatment'],
             '/other/health': ['/other/health/malady',
              '/other/health/treatment'],
             '/location/transit': ['/location/transit/railway',
              '/location/transit/road',
              '/location/transit/bridge'],
             '/other/event': ['/other/event/holiday',
              '/other/event/natural_disaster',
              '/other/event/election',
              '/other/event/violent_conflict',
              '/other/event/accident',
              '/other/event/sports_event',
              '/other/event/protest'],
             '/other/art': ['/other/art/writing',
              '/other/art/broadcast',
              '/other/art/music',
              '/other/art/stage',
              '/other/art/film'],
             '/other/product': ['/other/product/mobile_phone',
              '/other/product/computer',
              '/other/product/software',
              '/other/product/weapon',
              '/other/product/car'],
             '/organization': ['/organization/company',
              '/organization/military',
              '/organization/transit',
              '/organization/sports_team',
              '/organization/music',
              '/organization/sports_league',
              '/organization/company/news',
              '/organization/government',
              '/organization/company/broadcast',
              '/organization/education',
              '/organization/stock_exchange',
              '/organization/political_party'],
             '/location': ['/location/transit/railway',
              '/location/transit',
              '/location/geograpy/island',
              '/location/structure/hospital',
              '/location/geography/body_of_water',
              '/location/celestial',
              '/location/structure/restaurant',
              '/location/geography/mountain',
              '/location/park',
              '/location/city',
              '/location/structure/airport',
              '/location/structure/hotel',
              '/location/geography/island',
              '/location/country',
              '/location/structure/theater',
              '/location/transit/road',
              '/location/transit/bridge',
              '/location/structure/sports_facility',
              '/location/geography',
              '/location/structure/government',
              '/location/structure'],
             '/other/language': ['/other/language/programming_language'],
             '/person': ['/person/artist/author',
              '/person/military',
              '/person/artist/music',
              '/person/athlete',
              '/person/artist/actor',
              '/person/legal',
              '/person/title',
              '/person/doctor',
              '/person/artist/director',
              '/person/artist',
              '/person/religious_leader',
              '/person/political_figure',
              '/person/coach'],
             '/other/living_thing': ['/other/living_thing/animal'],
             '/person/artist': ['/person/artist/author',
              '/person/artist/music',
              '/person/artist/actor',
              '/person/artist/director'],
             '/location/geography': ['/location/geography/body_of_water',
              '/location/geography/mountain',
              '/location/geography/island'],
             '/location/structure': ['/location/structure/hospital',
              '/location/structure/restaurant',
              '/location/structure/airport',
              '/location/structure/hotel',
              '/location/structure/theater',
              '/location/structure/sports_facility',
              '/location/structure/government']}

class Vocab:
    def __init__(self, word2id_file, type2id_file, alpha):
        df_word2id    = pd.read_csv(word2id_file, names=["token"], dtype=str, keep_default_na=False)
        df_type2id    = pd.read_csv(type2id_file, names=["type"])
        self.idx_to_token, self.token_to_idx = [], dict()
        self.idx_to_type, self.type_to_idx  = [], dict()
        for token in df_word2id.token:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token)-1
        for t in df_type2id.type:
            self.idx_to_type.append(t)
            self.type_to_idx[t] = len(self.idx_to_type)-1
        self.prior = self.create_prior() # 子->父
        self.tune = self.create_prior(alpha=alpha).T # 父 -> 子
        
    def create_prior(self, alpha=1.0):
        num_types = len(self.idx_to_type)
        type_matrix = torch.zeros(num_types, num_types)
        for key in self.type_to_idx.keys():
            tmp = torch.zeros(num_types)
            tmp[self.type_to_idx[key]] = 1.0
            if key in type_dict.keys():
                for child_type in type_dict[key]:
                    tmp[self.type_to_idx[child_type]] = alpha
            type_matrix[:, self.type_to_idx[key]] = tmp
        return type_matrix
            

class Attention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, num_hiddens, dropout=0, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = nn.Linear(num_hiddens, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H):     # [batch_size*2, hidden], [batch_size, 2, hidden]              
        M = torch.tanh(H)    # 
        alpha = nn.functional.softmax(self.W(M), dim=-1).unsqueeze(-1)   #   [batch_size, 2, 1, 1]
        r = M.unsqueeze(-1) * self.dropout(alpha) #[batch_size, 2, hidden, 1]
        return r.squeeze(-1)
    
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, data_dir):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            data_dir)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, data_dir):
        idx_to_token, idx_to_vec = ['<unk>'], []
#         data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r',encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx)
            for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)