import jieba.analyse
str = "基于词图模型的关键词提取算法TextRank"
result = jieba.analyse.extract_tags(str,withWeight=True)
print(result)
