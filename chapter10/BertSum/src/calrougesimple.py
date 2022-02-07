#coding:utf-8
from rouge import Rouge
a = ["the group included four children , turkish official says<q>turkish military did n't say what group 's intent was<q>uk foreign office says it is trying to get information from turkish officials"]  # 预测摘要 （可以是列表也可以是句子）
b = ["the nine were arrested at the turkey-syria border , the turkish military says<q>it did n't say why the group allegedly was trying to get into Syria"] #真实摘要
 
'''
f:F1值  p：查准率  R：召回率
'''
rouge = Rouge()
rouge_score = rouge.get_scores(a, b)
print(rouge_score[0]["rouge-1"])
print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])
