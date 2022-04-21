from tqdm import tqdm
import json
import nltk

from dataclasses import dataclass

from typing import List, Dict

import torch

from transformers import PreTrainedTokenizer, BasicTokenizer
from transformers.tokenization_utils import _is_whitespace, _is_punctuation, _is_control

from torch.utils.data import Dataset, TensorDataset

class SQUADExample:
    def __init__(self,
                 qid: str,
                 exp_id: int,
                 title: str,
                 question_text: str,
                 context_text: str,
                 answer_text: str,
                 answer_start_index: int):
        self.qid = qid
        self.exp_id = exp_id
        self.title = title
        self.context_text = context_text
        self.question_text = question_text
        self.answer_text = answer_text
        self.answer_start_index = answer_start_index
        self.doc_tokens = []  # 新增
        self.char_to_word_offset = [] # 新增
        
        # raw_doc_tokens = customize_tokenizer(context_text, True) # 
        raw_doc_tokens = [token.replace("''", '"').replace("``", '"').lower() for token in nltk.word_tokenize(context_text)]
        k = 0
        temp_word = ""
        for char in self.context_text:
            if _is_whitespace(char):
                self.char_to_word_offset.append(k - 1)
                continue
            else:
                temp_word += char
                self.char_to_word_offset.append(k)
            if temp_word.lower() == raw_doc_tokens[k]: 
                self.doc_tokens.append(temp_word)
                temp_word = ""
                k += 1
        
        # assert k == len(raw_doc_tokens)
                
        if answer_text is not None:  # if for training
            answer_offset = context_text.index(answer_text)
            answer_length = len(answer_text) 
            self.start_position = self.char_to_word_offset[answer_offset]
            self.end_position = self.char_to_word_offset[answer_offset + answer_length - 1]
        
    def __repr__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}: {value}\n"
        return f"<{self.__class__}>"

    def copy(self):
        return copy.deepcopy(self)
        
def read_examples(file):
    with open(file, "r", encoding="utf-8") as file: # 读取json文件
        original_data = json.load(file)["data"]
    examples = []
    exp_idx = 0
    for entry in tqdm(original_data, disable=True):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qid = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                answer_start_index = qa['answers'][0]['answer_start']
                examples.append(SQUADExample(qid=qid, exp_id=exp_idx, title=title, context_text=context, question_text=question_text, \
                                            answer_text=answer_text, answer_start_index=answer_start_index))
                exp_idx = exp_idx + 1
    return examples

@dataclass
class SQUADFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    unique_id: int
    start_position: List[int]
    end_position: List[int]

def convert_examples_to_features(example_list: List[SQUADExample], tokenizer: PreTrainedTokenizer, is_training=True):
    feature_list = []
    i = 0
    for example in tqdm(example_list, disable=True):
        i += 1
        #if i % 3000 == 0:
            #print(i)
        query_tokens = tokenizer.tokenize(example.question_text)
        truncated_query = tokenizer.encode(query_tokens, add_special_tokens=False, max_length=24, truncation=True)
        encoded_dict = tokenizer.encode_plus(
            truncated_query, # 剪裁后的query_tokens
            example.doc_tokens,
            max_length=512,
            return_overflowing_tokens=True,
            padding="max_length",
            truncation="only_second", # 表示只截断第二个
            return_token_type_ids=True,
        )
        if is_training:
            curr_start_pos = [0 for i in range(512)] # 为了应用BCELoss
            curr_end_pos = [0 for i in range(512)]
            if example.start_position < 512 and example.end_position < 512:
                curr_start_pos[example.start_position] = 1
                curr_end_pos[example.end_position] = 1
            else:
                curr_start_pos[0] = curr_end_pos[0] = 1
            start_position = curr_start_pos
            end_position   = curr_end_pos 
        else:
            start_position = end_position = None
        feature_list.append(SQUADFeature(
                input_ids=encoded_dict["input_ids"],
                attention_mask=encoded_dict["attention_mask"],
                token_type_ids=encoded_dict["token_type_ids"],
                unique_id=example.exp_id,
                start_position=start_position,
                end_position=end_position,
        ))
    #print(i)
    return feature_list

def convert_features_to_dataset(features: List[SQUADFeature], is_training: bool) -> Dataset:
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    start_positions = torch.tensor([f.start_position for f in features], dtype=torch.float)
    end_positions = torch.tensor([f.end_position for f in features], dtype=torch.float)
    dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            start_positions,
            end_positions
        )
    return dataset
         
def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
