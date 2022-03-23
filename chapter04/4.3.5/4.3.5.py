import numpy as np
from numpy import *
import json


# Viterbi算法求测试集最优状态序列
def Viterbi(sentence, array_pi, array_a, array_b, STATES):
    weight = [{}]  # 动态规划表
    path = {}

    if sentence[0] not in array_b['B']:
        for state in STATES:
            if state == 'S':
                array_b[state][sentence[0]] = 0
            else:
                array_b[state][sentence[0]] = -3.14e+100

    for state in STATES:
        weight[0][state] = array_pi[state] + array_b[state][sentence[0]]
        # weight[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
        path[state] = [state]

    # 置分词开始和结束标志
    for state in STATES:
        if state == 'B':
            array_b[state]['begin'] = 0
        else:
            array_b[state]['begin'] = -3.14e+100
    for state in STATES:
        if state == 'E':
            array_b[state]['end'] = 0
        else:
            array_b[state]['end'] = -3.14e+100

    for i in range(1, len(sentence)):
        weight.append({})
        new_path = {}

        for state0 in STATES:  # state0表示sentence[i]的状态
            items = []
            for state1 in STATES:  # states1表示sentence[i-1]的状态
                # if sentence[i] not in array_b[state0]:  # sentence[i]为没有在观测矩阵array_b[state0]中出现的字符
                #     # sentence[i-1]为没有在观测矩阵array_b[state0]中出现的字符
                #     if sentence[i - 1] not in array_b[state0]:
                #         prob = weight[i - 1][state1] + array_a[state1][state0] + array_b[state0]['end']
                #     else:
                #         prob = weight[i - 1][state1] + array_a[state1][state0] + array_b[state0]['begin']
                # else:
                #     # 计算每个字符对应STATES的概率
                #     prob = weight[i - 1][state1] + array_a[state1][state0] + array_b[state0][sentence[i]]
                prob = weight[i - 1][state1] + array_a[state1][state0] + array_b[state0][sentence[i]]
                items.append((prob, state1))
            best = max(items)
            weight[i][state0] = best[0]
            new_path[state0] = path[best[1]] + [state0]
        path = new_path

    prob, state = max([(weight[len(sentence) - 1][state], state) for state in STATES])
    return path[state]


# 根据状态序列进行分词
def tag_seg(sentence, tag):
    word_list = []
    start = -1
    started = False

    if len(tag) != len(sentence):
        return None
    if len(tag) == 1:
        word_list.append(sentence[0])  # 语句只有一个字，直接输出
    else:
        if tag[-1] == 'B' or tag[-1] == 'M':  # 最后一个字状态不是'S'或'E'则修改
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'E'
            else:
                tag[-1] = 'S'
        for i in range(len(tag)):
            if tag[i] == 'S':
                # if started:
                #     started = False
                #     word_list.append(sentence[start:i])
                word_list.append(sentence[i])
            elif tag[i] == 'B':
                if started:
                    word_list.append(sentence[start:i])
                start = i
                started = True
            elif tag[i] == 'E':
                started = False
                word = sentence[start:i + 1]
                word_list.append(word)
            elif tag[i] == 'M':
                continue
    return word_list


if __name__ == '__main__':
    # trainset = open('CTBtrainingset.txt', encoding='utf-8')  # 读取训练集
    # testset = open('CTBtestingset.txt', encoding='utf-8')  # 读取测试集
    # output = {"states_matrix": array_A, "observation_matrix": array_B, "init_states": array_Pi}
    # json.dump(output, open('hmm_states.txt', "w", encoding="utf8"), indent=2, ensure_ascii=False)
    # 加载由训练集得到的统计数据
    pramater = json.load(open('hmm_states.txt', encoding='utf-8'))
    array_A = pramater['states_matrix']   # 状态转移概率矩阵
    array_B = pramater['observation_matrix']   # 观测状态概率矩阵
    array_Pi = pramater['init_states']  # 初始状态概率分布
    STATES = ['B', 'M', 'E', 'S']

    test = "中国游泳队在东京奥运会上取得了优异的成绩"
    tag = Viterbi(test, array_Pi, array_A, array_B, STATES)  # 使用维特比算法进行序列标注
    print(tag)
    seg = tag_seg(test, tag)   # 由标签结果进行分词
    print('/ '.join(seg))

