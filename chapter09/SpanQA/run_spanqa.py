import pickle
import gzip

import logging
import argparse

from collections import Counter, OrderedDict

import torch
from torch import nn

from os.path import join

from data_process import *

from transformers import ElectraModel, ElectraConfig,  ElectraTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from SpanQA import SpanQA, compute_loss

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_data(file_path):
    return pickle.load(gzip.open(file_path, "rb"))

def get_tokens(token_ids, starts, ends):
    tokens = []
    assert len(token_ids) == len(starts)
    for i in range(len(token_ids)):
        tokens.append(token_ids[i][starts[i]:ends[i]])
    return tokens

def get_em_scores(start_predicts, end_predicts, start_labels, end_labels):
    judge_start = [start_predicts[i] == start_labels[i] for i in range(len(start_predicts))]
    judge_end   = [end_predicts[i] == end_labels[i] for i in range(len(end_predicts))]
    return torch.tensor([judge_start[i] == True and judge_end[i] == True for i in range(len(start_predicts))]).sum().item() / len(start_predicts)

def get_f1_scores(token_ids, start_predicts, end_predicts, start_labels, end_labels):
    gold_toks = get_tokens(token_ids, start_labels, end_labels)
    pred_toks = get_tokens(token_ids, start_predicts, end_predicts)
    common = [Counter(gold) & Counter(pred) for gold, pred  in  zip(gold_toks, pred_toks)]
    num_same = sum(sum(com.values()) for com in common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / sum(len(toks) for toks in pred_toks)
    recall = 1.0 * num_same / sum(len(toks) for toks in gold_toks)
    #logging.info('precision: ', precision, num_same, sum(len(toks) for toks in pred_toks))
    #logging.info('recall: ', recall, num_same, sum(len(toks) for toks in gold_toks))
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(net, test_iter):
    start_poss_true, end_poss_true = [], []
    start_predicts, end_predicts = [], []
    test_token_ids = []

    for i, batch in enumerate(test_iter):
        net.eval()
        batch = tuple(t.to('cuda:1') for t in batch)   # to cuda
        with torch.no_grad():
            inputs = {
                "input_ids" : batch[0],
                "attention_mask" : batch[1],
                "token_type_ids" : batch[2],
            }
            start_logits, end_logits = net(**inputs)
        start_poss_true = start_poss_true + [p.index(1) for p in batch[3].tolist()]
        end_poss_true   = end_poss_true   + [p.index(1) for p in batch[4].tolist()]
        start_predicts  = start_predicts  + start_logits.argmax(-1).tolist()
        end_predicts    = end_predicts    + end_logits.argmax(-1).tolist()
        test_token_ids = test_token_ids + batch[0].tolist()
    assert len(start_predicts) == len(start_poss_true)
    em_score = get_em_scores(start_predicts, end_predicts, start_poss_true, end_poss_true)
    f1_score = get_f1_scores(test_token_ids, start_predicts, end_predicts, start_poss_true, end_poss_true)
    return {'EM' : em_score, 'F1' : f1_score}

def train_batch(net, batch, trainer, scheduler):
    inputs = {
        "input_ids" : batch[0],
        "attention_mask" : batch[1],
        "token_type_ids" : batch[2],
     }

    net.train()
    try: 
        start_logits, end_logits = net(**inputs) # 模型计算
        loss = compute_loss(start_logits, end_logits, batch[3], batch[4])
        loss.backward()
    except RuntimeError:
        logging.error(start_logits.shape, end_logits.shape, batch[3].shape, batch[4].shape)
        logging.error(start_logits, end_logits, batch[3], batch[4])
    trainer.step()
    scheduler.step()
    net.zero_grad()
    return loss.item()

def train_epoch(net, train_iter, test_iter, trainer, scheduler, num_epochs):
    num_batches = len(train_iter)
    test_acc = 0
    net.zero_grad()
    for epoch in range(int(num_epochs)):
        logging.info(f'开始第{epoch+1}轮训练')
        for i, batch in enumerate(train_iter):
            batch = tuple(t.to('cuda:1') for t in batch)   # to cuda
            l = train_batch(net, batch, trainer, scheduler)
            if (i + 1) % 200 == 0 or i == num_batches - 1:
                logging.info(f'loss: {l:.6f}')
            if (i + 1) % 1000 == 0 or i == num_batches - 1:
                results = evaluate(net, test_iter)
                logging.info(results)

    results = evaluate(net, test_iter)
    logging.info(f'Final: {results}')
    model_to_save = net.module if hasattr(net, "module") else net
    torch.save(model_to_save, './output_spanqa/checkpoint.bin')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to Electra model."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        required=True,
        type=str,
        help="The directory which contain the generated train-set examples and features file."
    )
    parser.add_argument("--device", default="cuda", type=str, help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training and evaluating.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. "
             "Longer will be truncated, and shorter will be padded."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer will be truncated to the length."
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    #electra_path = 'C:\\Users\\L0phTg\\work\\nlp\\bert-models\\google-electra-base-discriminator'
    config = ElectraConfig.from_pretrained(args.model_path)
    tokenizer = ElectraTokenizer.from_pretrained(args.model_path)
    pretrain_model = ElectraModel.from_pretrained(args.model_path, config=config)

    # 加载数据
    logging.info("加载数据...")
    # 1. 读取examples
    train_examples = read_examples(join(args.data_dir, 'train-v1.1.json'))
    dev_examples = read_examples(join(args.data_dir, 'dev-v1.1.json'))
    # 2. 得到features
    train_features = convert_examples_to_features(train_examples, tokenizer, is_training=True)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, is_training=True)
    logging.info(f'训练集:{len(train_features)}, 测试集:{len(dev_features)}')
    # 3. 转换为tensor
    train_dataset = convert_features_to_dataset(train_features, is_training=True)
    dev_dataset = convert_features_to_dataset(dev_features, is_training=True)

    # 加载模型
    logging.info("加载数据完成.")

    # 训练集
    train_batch_size = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    # 测试集
    dev_batch_size = args.batch_size
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=dev_batch_size)
    # 模型定义
    net = SpanQA(pretrain_model)
    net.to('cuda:1')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    trainer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    num_epochs = args.num_train_epochs
    t_total = len(train_dataloader) // num_epochs
    scheduler = get_linear_schedule_with_warmup(
        trainer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # 训练
    logging.info("开始训练...")
    train_epoch(net, train_dataloader, dev_dataloader, trainer, scheduler, num_epochs)

if __name__ == '__main__':
    main()
