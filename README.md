# BLSA-CRF
使用BLSA-CRF进行中文命名实体识别。突出一个词：**方便**。

数据和模型下载地址：https://cowtransfer.com/s/3a155b702dfa42 点击链接查看 [ BLSA-CRF ] ，或访问奶牛快传 cowtransfer.com 输入传输口令 hmpdf8 查看；

## 问题勘验
- TypeError: init() got an unexpected keyword argument 'batch_first'： `pip install pytorch-crf==0.7.2`

# 更新
[2025/01/03]

- 添加分布式训练，具体参考 
`transformers_trainer.py`以及启动脚本`train.sh`
- 添加转换onnx，具体参考convert_onnx/文件夹
- 添加支持triton推理，具体参考convert_triton/文件夹


# 依赖

```python
scikit-learn==1.1.3 
scipy==1.10.1 
seqeval==1.2.2
transformers==4.27.4
pytorch-crf==0.7.2
```

# 目录结构

```python
--checkpoint：模型和配置保存位置
--model_hub：预训练模型
----chinese-bert-wwm-ext:
--------vocab.txt
--------pytorch_model.bin
--------config.json
--data：存放数据
----dgre
--------ori_data：原始的数据
--------ner_data：处理之后的数据
------------labels.txt：标签
------------train.txt：训练数据
------------dev.txt：测试数据
--config.py：配置
--model.py：模型
--process.py：处理ori数据得到ner数据
--predict.py：加载训练好的模型进行预测
--main.py：训练和测试
```

# 说明

这里以dgre数据为例，其余数据类似。

```python
1、去https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main下载相关文件到chinese-bert-wwm-ext下。

2、在process.py里面定义将ori_data里面的数据处理得到ner_data下的数据

3、在config.py里面定义一些参数，比如：
--max_seq_len：句子最大长度，GPU显存不够则调小。
--epochs：训练的epoch数
--train_batch_size：训练的batchsize大小，GPU显存不够则调小。
--dev_batch_size：验证的batchsize大小，GPU显存不够则调小。
--save_step：多少step保存模型
其余的可保持不变。

4、在main.py里面修改data_name为数据集名称。需要注意的是名称和data下的数据集名称保持一致。最后运行：python main.py

5、在predict.py修改data_name并加入预测数据，最后运行：python predict.py
```


由于这几个项目的代码结构都差不多，而且都和信息抽取相关，就一起放在这。

- [BERT-BILSTM-CRF](https://github.com/taishan1994/BERT-BILSTM-CRF)：中文实体识别
- [BERT-Relation-Extraction](https://github.com/taishan1994/BERT-Relation-Extraction)：中文关系抽取
- [BERT-ABSA](https://github.com/taishan1994/BERT-ABSA)：中文方面级情感分析
- [BERT-Event-Extraction](https://github.com/taishan1994/BERT-Event-Extraction) 中文事件抽取
