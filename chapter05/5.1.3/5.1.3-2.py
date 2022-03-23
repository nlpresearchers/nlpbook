#导入相关模块
from textrank4zh import TextRank4Keyword
#定义要提取的文本
if __name__ == '__main__':
    text = ("燕山大学是河北省人民政府、教育部、工业和信息化部、国家国防科技工业局四方共建的全国重点大学，河北省重点支持的国家一流大学和世界一流学科建设高校，北京高科大学联盟成员。")
    #关键词提取
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=5)
    print('关键词：')
    for item in tr4w.get_keywords(10, word_min_len=1):
         print(item['word'], item['weight'])
