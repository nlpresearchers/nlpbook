#词向量
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

# 读入训练集文件
data = pd.read_csv('train.csv')
# 转字符串数组
corpus = data['comment'].values.astype(str)
# 分词，再重组为字符串数组
corpus = [jieba.lcut(corpus[index]
                          .replace("，", "")
                          .replace("!", "")
                          .replace("！", "")
                          .replace("。", "")
                          .replace("~", "")
                          .replace("；", "")
                          .replace("？", "")
                          .replace("?", "")
                          .replace("【", "")
                          .replace("】", "")
                          .replace("#", "")
                        ) for index in range(len(corpus))]
# 词向量模型训练
model = Word2Vec(corpus, sg=0, vector_size=300, window=5, min_count=3, workers=4)
#模型显示
print('模型参数：',model,'\n')
#最匹配
print('最匹配的词是：',model.wv.most_similar(positive=['点赞', '不错'], negative=['难吃']),'\n')
#最不匹配
#print('最不匹配的词是：',model.wv.doesnt_match("点赞 好吃 支持 难吃".split()),'\n')
#语义相似度
print('相似度为=',model.wv.similarity('推荐','好吃'),'\n')
#坐标返回
print(model.wv.__getitem__('地道'))
