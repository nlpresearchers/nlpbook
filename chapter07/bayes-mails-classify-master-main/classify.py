import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
all_words = []
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


top_words = get_top_words(100)
# 构建词-个数映射表
vector = []
for words in all_words:
    '''
    words:
    ['国际', 'SCI', '期刊', '材料', '结构力学', '工程', '杂志', '国际', 'SCI', '期刊', '先进', '材料科学',
    '材料', '工程', '杂志', '国际', 'SCI', '期刊', '图像处理', '模式识别', '人工智能', '工程', '杂志', '国际',
    'SCI', '期刊', '数据', '信息', '科学杂志', '国际', 'SCI', '期刊', '机器', '学习', '神经网络', '人工智能',
    '杂志', '国际', 'SCI', '期刊', '能源', '环境', '生态', '温度', '管理', '结合', '信息学', '杂志', '期刊',
    '网址', '论文', '篇幅', '控制', '以上', '英文', '字数', '以上', '文章', '撰写', '语言', '英语', '论文',
    '研究', '内容', '详实', '方法', '正确', '理论性', '实践性', '科学性', '前沿性', '投稿', '初稿', '需要',
    '排版', '录用', '提供', '模版', '排版', '写作', '要求', '正规', '期刊', '正规', '操作', '大牛', '出版社',
    '期刊', '期刊', '质量', '放心', '检索', '稳定', '邀请函', '推荐', '身边', '老师', '朋友', '打扰', '请谅解']
    '''
    word_map = list(map(lambda word: words.count(word), top_words))
    '''
    word_map:
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    vector.append(word_map)
vector = np.array(vector)
# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)
model = MultinomialNB()
model.fit(vector, labels)
def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = np.array(
        tuple(map(lambda word: words.count(word), top_words)))
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))
