#（1）导入所需的模块
from transformers import BertModel
from transformers import BertTokenizer
import torch

#（2）加载预训练模型
# 加载模型，将预先下载好的模型存入某一目录下（如D:\pythonProject/bert）
model = BertModel.from_pretrained('./bert')
# 加载分词器
tokenizer = BertTokenizer.from_pretrained('./bert')

#（3）获取词向量
text = "中"
# 获取词在词表中的索引, add_special_tokens参数用于控制是否添加[SEP]、[CLS]等特殊token
input_id = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
# 获取此索引对应的词向量
out_vec = model(input_id)[0]
print(out_vec)
