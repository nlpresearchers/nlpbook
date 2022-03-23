def FMM(dict, sentence):
    """
    正向最大匹配（FMM）
    dict: 词典
    sentence: 句子
    """

    fmmresult = []
    # 词典中最长词长度
    max_len = max([len(item) for item in dict])
    start = 0
    # FMM为正向，start从初始位置开始，指向结尾即为结束
    while start != len(sentence):
        # index的初始值为start的索引+词典中元素的最大长度或句子末尾
        index = start + max_len
        if index > len(sentence):
            index = len(sentence)
        for i in range(max_len):
            # 当分词在字典中时或分到最后一个字时，将其加入到结果列表中
            if (sentence[start:index] in dict) or (len(sentence[start:index]) == 1):
                # print(sentence[start:index], end='/')
                fmmresult.append(sentence[start:index])
                # 分出一个词，start设置到index处
                start = index
                break
            # 正向时index每次向句尾挪一位
            index += -1
    return fmmresult

def RMM(dict, sentence):
    """
    逆向最大匹配（RMM）
    dict: 词典
    sentence: 句子
    """

    rmmresult = []
    # 词典中最长词长度
    max_len = max([len(item) for item in dict])
    start = len(sentence)
    # RMM为逆向，start从末尾位置开始，指向开头位置即为结束
    while start != 0:
        # 逆向时index的初始值为start的索引-词典中元素的最大长度或句子开头
        index = start - max_len
        if index < 0:
            index = 0
        for i in range(max_len):
            # 当分词在字典中时或分到最后一个字时，将其加入到结果列表中
            if (sentence[index:start] in dict) or (len(sentence[index:start]) == 1):
                # print(sentence[index:start], end='/')
                rmmresult.insert(0, sentence[index:start])
                # 分出一个词，start设置到index处
                start = index
                break
            # 逆向时index每次向句头挪一位
            index += 1
    return rmmresult

def BM(dict, sentence):
    # res1 与 res2 为FMM与RMM结果
    res1 = FMM(dict, sentence)
    res2 = RMM(dict, sentence)
    if len(res1) == len(res2):
        # FMM与RMM的结果相同时，取任意一个
        if res1 == res2:
            return res1
        else:
            # res1_sn 和 res2_sn 为两个分词结果的单字数量，返回单字较少的
            res1_sn = len([i for i in res1 if len(i) == 1])
            res2_sn = len([i for i in res2 if len(i) == 1])
            return res1 if res1_sn < res2_sn else res2
    else:
        # 分词数不同则取分出词较少的
        return res1 if len(res1) < len(res2) else res2


dict = ['今日', '阳光明媚', '光明', '明媚', '阳光', '我们', '在', '在野', '生动', '野生',
        '动物园', '野生动物园', '物', '园', '玩']
sentence = '在野生动物园玩'

print("the results of FMM :\n", FMM(dict, sentence), end="\n")

print("the results of RMM :\n", RMM(dict, sentence), end="\n")

print("the results of BM :\n", BM(dict, sentence))
