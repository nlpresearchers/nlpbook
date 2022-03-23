import json
from math import log10
class Bigram():
    #（1）定义函数__init__，用于初始化Bigram类。
    #在函数中，初始化词典、词频统计、标点符号等变量。核心代码如下：
    def __init__(self, punc, wordset, bigram_wordset):
      self.DICT = wordset   # 词典
      self.BIGRAM = bigram_wordset     # 词与相邻词的频率统计
      self.PUNC = punc     # 标点符号
      self.alpha = 2e-4      # 用于未登录词
    #（2）定义函数_forwardSplitSentence，用于获取sentence所有可能的切分方案，是实现本分词算法的核心功能函数之一。在函数中，变量sentence表示待切分文本，word_max_len代表最大词长，split_groups用于存储所有方案，最终要返回切分方案列表。核心代码如下：
    # 用于获取sentence所有可能的切分方案
    def _forwardSplitSentence(self, sentence, word_max_len=5):
      #'''
      #前向切分
      #:param sentence:待切分文本
      #:param word_max_len:最大词长
      #:return:切分方案列表
      #'''
      # 所有可能的切分方案
        split_groups = []  # 用于存储所有方案
        sentence = sentence.strip()  # 去除空格
        sentence_len = len(sentence)
        if sentence_len < 2:       # sentence只有一个词时，不用切分
            return [[sentence]]
        # 取待划分句子长度和最大词长度之中的较小值
        range_len = [sentence_len,word_max_len][sentence_len > word_max_len]
        current_groups = []    # 保存当前二切分结果
        single_cut = True     # 是否需要从第一个字后进行切分
        for i in range(1, range_len)[::-1]: # 反向取值，i依次取range_len-1,range_len-2,...,1
            #子词串在词典中存在，进行二分切分
            if self.DICT.__contains__(sentence[:i]) and i != 1: # 逆向切分，不从第一个字后切分
                current_groups.append([sentence[:i], sentence[i:]])
                single_cut = False
        # 没有在字典词组中匹配到，或者第2个字和第3个字构成词
        if single_cut or self.DICT.__contains__(sentence[1:3]):
            current_groups.append([sentence[:1], sentence[1:]])  # 从第1个字后切分
        # 词长为2时，为未登录词的概率较大，保留“为词”的可能性
        if sentence_len == 2:
            current_groups.append([sentence])
        # 对每一个切分，递归组合
        for one_group in current_groups:  # one_group为一种二划分结果
            if len(one_group) == 1:      # 划分集合中只有一个词，无需再划分
                split_groups.append(one_group)
                continue
            # 对二划分的后一个分片进行再次划分，得到所有划分方案
            for child_group in self._forwardSplitSentence(one_group[1]):
                child_group.insert(0, one_group[0])  # 在方案前添加二划分的前一个分片
                split_groups.append(child_group)    # 加入到结果集
        return split_groups
    #（3）定义函数getPValue，用于查询二元概率，是实现本分词算法的另一核心功能函数。
    #在函数中，front_word为前向词，word是当前词，返回 P(Wi|Wi-1) 。核心代码如下：
    def getPValue(self, front_word, word):
      #'''
      #查询二元概率
      #:param front_word: 前向词
      #:param word: 当前词
      #:return: P(Wi|Wi-1)
      #'''
        if front_word in self.BIGRAM and word in self.BIGRAM[front_word]:
             return self.BIGRAM[front_word][word]
        return self.alpha
    #（4）定义函数_maxP，用于计算最大概率的切分组合。
    #在函数中，变量sentence表示待切分句子，word_max_len为最大不切分词长，返回最优切分方案。核心代码如下：
    def _maxP(self, sentence, word_max_len=5):
       # 获取切分组合
        split_words_group =self._forwardSplitSentence(sentence, word_max_len=word_max_len)
        max_p = -99999
        best_split = []   # 存放结果
        value_dict = {}  # 存放已经计算过概率的子序列
        value_dict[u'<start>'] = dict()  # u'<start>'是句子第一个词的前向词
        for split_words in split_words_group[::-1]: # 取方案
            words_p = 0  # 记录概率
            try:
                for i in range(len(split_words)):
                    word = split_words[i]
                    if i == 0:    # 第一个词，特殊处理
                        if word not in value_dict[u'<start>']:
                            # 获取该词在（前向词|词）中的概率
                            value_dict[u'<start>'][word] =log10(self.getPValue(u'<start>', word))
                        words_p += value_dict[u'<start>'][word]  # 概率累计
                        continue
                    front_word = split_words[i - 1]  # 找到前向词
                    if front_word not in value_dict:  # 前向词不在字典中
                        value_dict[front_word] = dict()  # 将前向词插入
                    if word not in value_dict[front_word]:  # 前向词中没有当前词的概率
                        value_dict[front_word][word] = log10(self.getPValue(front_word, word))# 赋值
                    words_p += value_dict[front_word][word]   # 每个p(wi|wi-1)求和#
                    #if words_p < max_p
                    #    break
            except ValueError:
                print("Failed to calc:\:late maxP.")
            if words_p > max_p:     # 获取累加概率最高的划分方案
                max_p = words_p
                best_split = split_words
        return best_split
    #（5）定义函数segment，是分词的调用入口。相比英文，中文也具有天然的分割词，即各种标点符号，本函数将输入文本按照标点切分成多个文本，再对其分别进行分词。
    #在函数中，定义变量sentence来指定出待分词文本，return返回最终的切分序列。核心代码如下：
    def segment(self, sentence):
    #'''
    #分词调用入口
    #:param sentence:待切分句子
    #:return:切分词序列
    #'''
        words = []
        sentences = []
        # 若含有标点，以标点分割
        start = -1
        for i in range(len(sentence)):
            if sentence[i] in self.PUNC:
                sentences.append(sentence[start + 1:i])
                sentences.append(sentence[i])
                start = i
        if not sentences:  # 不含标点
            sentences.append(sentence)
        for sent in sentences:
            words.extend(self._maxP(sent))
        return words
#（6）定义函数getPunciton，用于获取标点。
#在函数中，定义变量file_name来指定文本路径，is_save表示是否保存，save_file表示保存路径。核心代码如下：
def getPunciton(file_name='199801.txt', is_save=True, save_file='punction.txt'):
    #'''
    #获取标点
    #:param file_name: 文本路径
    #:param is_save: 是否保存
    #:param save_file: 保存路径
    #:return:
    #'''
        punction = set(['[', ']'])
        with open(file_name, 'rb') as f:
            for line in f:
                content = line.decode('gbk').strip().split()
                # 去掉第一个词“19980101-01-001-001/m”
                for word in content[1:]:
                    if word.split(u'/')[1] == u'w':
                        punction.add(word.split(u'/')[0])
        if is_save:
            # punction
            with open(save_file,"w",encoding='utf-8') as f:
                f.write('\n'.join(punction))
        return punction
#（8）定义函数toWordSet，用于获取词典。
#在函数中，定义变量file_name来指定文本路径，is_save表示是否保存，save_file表示保存路径，return返回词典。核心代码如下：
def toWordSet(file_name='199801.txt', is_save=True, save_file='wordSet.json'):
    #'''
    #获取词典
    #:param file_name: 文本路径
    #:param is_save: 是否保存
    #:param save_file: 保存路径
    #:return:
    #'''
        word_dict = {}
        with open(file_name, 'rb') as f:
            for line in f:
                content = line.decode('gbk').strip().split()
                # 去掉第一个词“19980101-01-001-001/m”
                for word in content[1:]:
                    word = word.split(u'/')[0]
                    if not word_dict.__contains__(word):
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1
        if is_save:
            # 保存wordSet以复用
            with open(save_file,'w',encoding='utf-8') as f:
                json.dump(word_dict, f, ensure_ascii=False, indent=2)
        print("successfully get word dictionary!")
        print("the total number of words is:{0}".format(len(word_dict.keys())))
        return word_dict
#（9）调用函数进行分词
#核心代码如下：
if __name__ == '__main__':
    # 加载符号
    punc = getPunciton()
    # 加载词典
    word_set = json.load(open('wordSet.json','r',encoding='utf-8'))
    # 加载Bigram词表
    bigram_wordset=json.load(open('word_distri.json','r',encoding='utf-8'))
    bigram = Bigram(punc, word_set, bigram_wordset)
    s = '我喜欢观赏日出'
    print(bigram.segment(s))
