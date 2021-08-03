import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
device=torch.device('cuda:0')

# 为CPU中设置种子，生成随机数
torch.manual_seed(1)

# 得到最大值的索引
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 转移矩阵，transaction[i][j]表示从label_j转移到label_i的概率，虽然是随机生成的，但是后面会迭代更新
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 设置任何标签都不可能转移到开始标签。设置结束标签不可能转移到其他任何标签
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # 随机初始化lstm的输入（h_0，c_0）
        self.hidden = self.init_hidden()

    # 随机生成输入的h_0,c_0
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # 求所有可能路径得分之和
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 输入：发射矩阵，实际就是LSTM的输出————sentence的每个word经LSTM后，对应于每个label的得分
        # 输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里以便自动反向传播
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # 当前层这一点的发射得分要与上一层所有点的得分相加，为了用加快运算，将其扩充为相同维度的矩阵
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # 前一层5个previous_tags到当前层当前tag_i的transition scors
                trans_score = self.transitions[next_tag].view(1, -1)
                # 前一层所有点的总得分 + 前一节点标签转移到当前结点标签的得分（边得分） + 当前点的发射得分
                next_tag_var = forward_var + trans_score + emit_score
                # 求和，实现w_(t-1)到w_t的推进
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 保存的是当前层所有点的得分
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 最后将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # 输入：id化的自然语言序列
        # 输出：序列中每个字符的Emission Score
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm模型的输出矩阵维度为（seq_len，batch，num_direction*hidden_dim）
        # 所以此时lstm_out的维度为（11,1,4）
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 把batch维度去掉，以便接入全连接层
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # 用一个全连接层将其转换为（seq_len，tag_size）维度，才能生成最后的Emission Score
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 输入：feats——emission scores；tag——真实序列标注，以此确定转移矩阵中选择哪条路径
        # 输出：真是路径得分
        score = torch.zeros(1)
        # 将START_TAG的标签3拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # 路径得分等于：前一点标签转移到当前点标签的得分 + 当前点的发射得分
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 最后加上STOP_TAG标签的转移得分，其发射得分为0，可以忽略
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 预测路径得分，维特比解码，输出得分与路径值
        backpointers = []

        # Initialize the viterbi variables in log space
        # B:0  I:1  O:2  START_TAG:3  STOP_TAG:4
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # 维特比解码的开始：一个START_TAG，得分设置为0，其他标签的得分可设置比0小很多的数
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var表示当前这个字被标注为各个标签的得分（概率）
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        # 遍历每个字，过程中取出这个字的发射得分
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            # 遍历每个标签，计算当前字被标注为当前标签的得分
            for next_tag in range(self.tagset_size):
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # forward_var保存的是之前的最优路径的值，然后加上转移到当前标签的得分，
                # 得到当前字被标注为当前标签的得分（概率）
                next_tag_var = forward_var + self.transitions[next_tag]
                # 找出上一个字中的哪个标签转移到当前next_tag标签的概率最大，并把它保存下载
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                # 把最大的得分也保存下来
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 然后加上各个节点的发射分数，形成新一层的得分
            # cat用于将list中的多个tensor变量拼接成一个tensor
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 得到了从上一字的标签转移到当前字的每个标签的最优路径
            # bptrs_t有5个元素
            backpointers.append(bptrs_t)

        # 其他标签到结束标签的转移概率
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        # 最终的最优路径得分
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        # 无需返回最开始的start标签
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        # 把从后向前的路径正过来
        best_path.reverse()
        return path_score, best_path

    # 损失函数
    def neg_log_likelihood(self, sentence, tags):
        # len(s)*5
        feats = self._get_lstm_features(sentence)
        # 规范化因子 | 配分函数 | 所有路径的得分之和
        forward_score = self._forward_alg(feats)
        # 正确路径得分
        gold_score = self._score_sentence(feats, tags)
        # 已取反
        # 原本CRF是要最大化gold_score - forward_score，但深度学习一般都最小化损失函数，所以给该式子取反
        return forward_score - gold_score

    # 实际上是模型的预测函数，用来得到一个最佳的路径以及路径得分
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 解码过程，维特比解码选择最大概率的标注路径

        # 先放入BiLstm模型中得到它的发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # 然后使用维特比解码得到最佳路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
# 标签一共有5个，所以embedding_dim为5
EMBEDDING_DIM = 5
# BILSTM隐藏层的特征数量，因为双向所以是2倍
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# 首先是用未训练过的模型随便预测一个结果
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))


# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):
    for sentence, tags in training_data:
        # 训练前将梯度清零
        optimizer.zero_grad()

        # 准备输入
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 前向传播，计算损失函数
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 反向传播计算loss的梯度
        loss.backward()
        # 通过梯度来更新模型参数
        optimizer.step()

# 使用训练过的模型来预测一个序列，与之前形成对比
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))