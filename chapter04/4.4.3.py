import jieba
#全模式
seg_list = jieba.cut("燕山大学源于哈尔滨工业大学，始建于1920年",cut_all=True)
print("全模式:", "/ ".join(seg_list))
#精确模式
seg_list = jieba.cut("燕山大学源于哈尔滨工业大学，始建于1920年",cut_all=False)
print("精确模式:", "/ ".join(seg_list))
#默认是精确模式
seg_list = jieba.cut("燕山大学源于哈尔滨工业大学，始建于1920年")
print("默认模式:",", ".join(seg_list))
#搜索引擎模式
seg_list = jieba.cut_for_search("燕山大学源于哈尔滨工业大学，始建于1920年")
print("搜素引擎模式:",", ".join(seg_list))
