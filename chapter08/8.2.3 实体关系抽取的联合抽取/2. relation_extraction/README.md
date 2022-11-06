本代码为基于序列标注的联合抽取算法，参考『2021语言与智能技术竞赛』- [关系抽取任务基线系统](https://aistudio.baidu.com/aistudio/competition/detail/65)
## 说明
本项目使用的模型与书中描述的模型一致，但序列标注方式存在不同，代码通俗易懂，适合入门练习。
## 环境依赖

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install --upgrade paddlenlp\>=2.0.0rc5
   ```

* 环境依赖

   Python的版本要求 3.6+，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 部分的内容


## 目录结构


以下是本项目主要目录结构及说明：

```text
relation_extraction/
├── data_loader.py # 加载数据
├── extract_chinese_and_punct.py # 文本数据预处理
├── README.md # 文档说明
├── re_official_evaluation.py # 评价脚本
├── run_duie.py # 模型训练脚本
├── train.sh # 启动脚本
└── utils.py # 效能函数
```
