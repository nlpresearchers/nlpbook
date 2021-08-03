import pandas as pd

import torch
from torch import nn
from d2l import torch as d2l

from util import Vocab, Attention, TokenEmbedding

def parse(df_data, num_classes, vocab):
    train_positions = torch.tensor([[x, y] for x, y in zip(df_data.p1, df_data.p2)])
    train_words     = df_data.words.apply(lambda x : [vocab.token_to_idx[tok] for tok in x.strip().split()])
    train_mentions = []
    for i in range(len(train_positions)):
        train_mentions.append(train_words[i][train_positions[i][0]:train_positions[i][1]])
    train_mentions_ext  = df_data.mentions.apply(lambda x : [vocab.token_to_idx[tok] for tok in x.strip().split()])
    train_types     = df_data.types.apply(lambda x : [vocab.type_to_idx[t] for t in x.split()]) # 多标签预测
    train_types_multiclass = torch.zeros(train_types.size, num_classes)
    for i in range(train_types.size):
        for idx in train_types[i]:
            train_types_multiclass[i][idx] = 1
    # truncate, words_size = 24, mention_size = 2, mentions_ext_size = 3
    train_words = torch.tensor([d2l.truncate_pad(words, 24, vocab.token_to_idx['<PAD>'])for words in train_words])
    train_mentions = torch.tensor([d2l.truncate_pad(mentions, 2, vocab.token_to_idx['<PAD>'])for mentions in train_mentions])
    train_mentions_ext = torch.tensor([d2l.truncate_pad(mentions, 4, vocab.token_to_idx['<PAD>'])for mentions in train_mentions_ext])
    return (train_words, train_mentions, train_mentions_ext), train_types_multiclass
    
def load_data(train_file, test_file, vocab):
    df_train = pd.read_csv(train_file, names=["p1", "p2", "words", "mentions", "types"])
    df_test  = pd.read_csv(test_file, names=["p1", "p2", "words", "mentions", "types"])
    
    train_data, train_label = parse(df_train, len(vocab.idx_to_type), vocab)
    test_data, test_label   = parse(df_test, len(vocab.idx_to_type), vocab)
    
    return train_data, train_label, test_data, test_label


# return [batch_size, embeds_size]
def average_encoder(words): # batch_size, 24, embeds_size
    return torch.sum(words, dim=1)/words.shape[1]

class NFETC(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, out_size,
                 **kwargs):
        super(NFETC, self).__init__(**kwargs)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.avg_encoder = average_encoder
        self.encoder1  = nn.LSTM(embed_size, num_hiddens, num_layers=2, dropout=0.5)
        self.encoder2  = nn.LSTM(embed_size, num_hiddens, num_layers=2, bidirectional=True, dropout=0.5)
        
        self.attention = Attention(num_hiddens, dropout=0.5)
        
        self.fc1       = nn.Linear(embed_size, num_hiddens)
        self.fc2       = nn.Linear(4*num_hiddens, out_size)
        
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(4, affine=False)
        
    def forward(self, inputs, tune):
        words = inputs[0]  
        mentions = inputs[1] 
        mentions_ext    = inputs[2]  
        bsz = words.shape[0] if words.dim() > 1 else 1
            
        mentions_embeds = self.embedding(mentions)            
        mentions_ext_embeds = self.embedding(mentions_ext.T)  
        words_embeds = self.embedding(words.T)  
        
        outputs1, _ = self.encoder1(mentions_ext_embeds)  
        outputs2, _ = self.encoder2(words_embeds) 
        outputs_added = torch.add(outputs2[0], outputs2[-1])
        
        ra = self.fc1(self.avg_encoder(mentions_embeds)).unsqueeze(1) 
        rl = outputs1[-1].unsqueeze(1)  
        rc = self.attention(outputs_added.reshape(bsz, -1, ra.shape[-1]))
        R = torch.cat((rc, ra, rl), dim=1) 

        h_drop = self.dropout(self.relu(R)) # NORMAL
        h_output = self.batch_norm(h_drop)
        scores = self.fc2(h_output.reshape(bsz, -1)) 

        proba = nn.functional.softmax(scores, dim=1) # HIER
        adjusted_proba = torch.mm(proba, tune)
        adjusted_proba = torch.clamp(adjusted_proba, 1e-10, 1.0)
        
        return adjusted_proba


def accuracy(preds, y):
    predictions = torch.argmax(preds, dim=1)
    type_path = vocab.prior[predictions]           # child -> parent路径, 路径准确度
    match_types = torch.mul(type_path, y)
    return match_types.sum()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for (words, mentions, mentions_ext, y,) in data_iter:
        X = (words, mentions, mentions_ext)
        if isinstance(X, list) or isinstance(X, tuple):
            # Required for BERT Fine-tuning (to be covered later)
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X, vocab.tune), y), y.sum())
    return metric[0] / metric[1]

def hier_loss(pred, y):
    target = torch.argmax(torch.mul(pred, y), dim=1)
    target_index = nn.functional.one_hot(target, len(vocab.idx_to_type))
    loss = -torch.sum(target_index *torch.log(pred), 1)
    return loss

def train_batch(net, X, y, loss, trainer, devices):
    X = [x.to(devices[0]) for x in X]
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X, vocab.tune)     # 模型计算
    
    l = loss(pred, y) # 计算损失
    l.sum().backward()# backward
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_epochs(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    num_batches = len(train_iter)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (words, mentions, mention_exts, types) in enumerate(train_iter):
            l, acc = train_batch(net, (words, mentions, mention_exts), types, loss, trainer,
                                      devices)
            metric.add(l, acc, types.shape[0], types.sum())

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'Final loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')

def main():
    # 加载训练数据
    print('load train data...')
    train_data, train_label, test_data, test_label = load_data('./data/ontonotes_large/train_ontonotes_large.csv', './data/ontonotes_large/test_ontonotes_large.csv', vocab)

    # 初始化模型
    embed_size, num_hiddens, batch_size, devices = 100, 128, 256, d2l.try_all_gpus()
    net = NFETC(len(vocab.idx_to_token), embed_size, num_hiddens, len(vocab.idx_to_type))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)

    # 加载词向量
    print('load embedding...')
    glove_embedding = TokenEmbedding('path\\models\\glove.6B.100d') # 指定glove的路径
    # glove_embedding = TokenEmbedding('/models/glove/glove.840B.300d')
    embeds = glove_embedding[vocab.idx_to_token]

    net.embedding.weight.data.copy_(embeds) # 设置embeds
    net.embedding.weight.requires_grad = False
    # 训练器参数
    lr, num_epochs = 0.0002, 20
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = hier_loss
    vocab.tune = vocab.tune.to(devices[0])
    vocab.prior = vocab.prior.to(devices[0])
    net.to(devices[0])
    # 训练数据迭代器
    train_iter = d2l.load_array((train_data[0], train_data[1], train_data[2], train_label), batch_size)
    test_iter  = d2l.load_array((test_data[0], test_data[1], test_data[2], test_label), batch_size)
    print('ready training...')
    train_epochs(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 加载参数
vocab = Vocab('./data/ontonotes_large/token2id.txt', './data/ontonotes_large/type2id.txt', alpha=0.3)
main()