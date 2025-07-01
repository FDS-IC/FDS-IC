# import os
# import torch
# import numpy as np
# from transformers import BertTokenizer
# from model import BertNer  # 请确保这是你自定义的模型实现
#
#
# class Args:
#     def __init__(self, bert_dir, num_labels, max_seq_len):
#         self.bert_dir = bert_dir
#         self.num_labels = num_labels
#         self.max_seq_len = max_seq_len
#
#
# class Predictor:
#     def __init__(self, model_dir):
#         # 设置设备
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 加载分词器
#         self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
#
#         # 标签列表（需和训练时一致）
#         self.label_list = [ "O", "B-O","I-O","B-B-cause","I-B-cause","B-I-cause","I-I-cause","B-B-location",
#                             "I-B-location","B-I-location","I-I-location","B-B-loss","I-B-loss","B-I-loss",
#                             "I-I-loss","B-B-time","I-B-time","B-I-time","I-I-time","B-B-trigger","I-B-trigger",
#                             "B-I-trigger","I-I-trigger"]
#         self.id2label = {i: label for i, label in enumerate(self.label_list)}
#
#         # 初始化参数对象
#         args = Args(bert_dir="hfl/chinese-bert-wwm-ext", num_labels=len(self.label_list), max_seq_len=128)
#         self.args = args
#
#         # 加载模型
#         self.model = BertNer(self.args)
#         self.model.load_state_dict(torch.load(f"./checkpoint/{model_dir}/pytorch_model_ner.bin", map_location=self.device))
#         self.model.to(self.device)
#         self.model.eval()
#
#     def ner_predict(self, text):
#         # 分词编码
#         encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding=False, add_special_tokens=True)
#         input_ids = encoding["input_ids"].to(self.device)
#         attention_mask = encoding["attention_mask"].to(self.device)
#
#         # 模型预测
#         with torch.no_grad():
#             logits = self.model(input_ids, attention_mask=attention_mask)
#
#         # 获取预测 ID
#         pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
#
#         # 获取原始 token（去掉特殊符号）
#         tokens = self.tokenizer.tokenize(text)
#         pred_ids = pred_ids[1:len(tokens) + 1]
#         pred_labels = [self.id2label.get(int(i), "O") for i in pred_ids]
#
#         # 实体抽取
#         result = {}
#         current_label = None
#         current_text = ""
#
#         for token, label in zip(tokens, pred_labels):
#             if label.startswith("B-"):
#                 if current_label:
#                     result.setdefault(current_label, []).append((current_text,))
#                 current_label = label[2:]
#                 current_text = token.lstrip("##")
#             elif label.startswith("I-") and current_label == label[2:]:
#                 current_text += token.lstrip("##")
#             else:
#                 if current_label:
#                     result.setdefault(current_label, []).append((current_text,))
#                     current_label = None
#                     current_text = ""
#
#         # 处理最后一个实体
#         if current_label and current_text:
#             result.setdefault(current_label, []).append((current_text,))
#
#         return result


import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r", encoding="utf-8") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin"), map_location="cpu"))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result

if __name__ == "__main__":
    data_name = "dgre"  # 可改为 "duie"
    predictor = Predictor(data_name)

    if data_name == "dgre":
        texts = [
            "山西太原一小区天然气管道泄漏，猛烈喷出数米高。2019年10月11日，太原迎新街一天然气管道泄漏，致使管道内的天然气猛烈喷涌而出。",
            "2019年8月20日晚上，在浙江打工的小张，因为下意识的一个动作，身受重伤。据了解，当天小张下班后，突然想起忘记关掉仓库里的煤气罐，于是就赶回仓库查看。走到煤气罐的旁边，小张赶紧把阀门关了。但是由于仓库黑灯瞎火，小张为了检查煤气罐是否关紧，于是就掏出了“打火机”！！！",
            "2019年10月18日凌晨0点左右，上海浦东归昌路凌河路交叉口的浦兴菜市场一面馆发生爆燃事故，据初步了解：不排除事发面馆内的液化气发生泄漏后发生爆燃事故，未造成人员伤亡。事故原因可能使用液化气操作不当。",
        ]
    elif data_name == "duie":
        texts = [
            "2024年6月13日，江苏南通。通州区川姜镇一沿街居民楼突然发生爆炸，通州区应急管理局工作人员称，事故原因可能是液化气瓶泄漏导致的燃爆，事故造成2人受伤，没有生命危险。",
            "2024年4月10日6时15分左右，苏家屯区轻飏小筑小区一居民家中发生液化气爆燃事故，造成4人受伤，均已第一时间送医，其中1人抢救无效死亡。事故原因用户操作原因造成液化气泄漏爆燃。",
        ]

    for text in texts:
        ner_result = predictor.ner_predict(text)
        print("文本 >>>>>", text)
        print("实体 >>>>>", ner_result)
        print("=" * 80)


# # import os
# # import json
# # import torch
# # import numpy as np
# # import torch
# # import numpy as np
# # from torch.nn.functional import softmax
# # from collections import namedtuple
# # from model import BertNer
# # from seqeval.metrics.sequence_labeling import get_entities
# # from transformers import BertTokenizer, BertModel
# #
# # # import os
# # # os.environ['CURL_CA_BUNDLE'] = ''
# #
# # tokenizer = BertTokenizer.from_pretrained("./model_hub/chinese-bert-wwm-ext")
# # model = BertModel.from_pretrained("./model_hub/chinese-bert-wwm-ext")
# # def get_args(args_path, args_name=None):
# #     with open(args_path, "r", encoding="utf-8", errors="replace") as fp:
# #         args_dict = json.load(fp)
# #     # 注意args不可被修改了
# #     args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
# #     return args
# #
# #
# # class Predictor:
# #
# #
# #     def __init__(self, data_name):
# #         self.data_name = data_name
# #         self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
# #         self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
# #         self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
# #         self.max_seq_len = self.ner_args.max_seq_len
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.ner_model = BertNer(self.ner_args)
# #
# #         # 加载模型权重，忽略不匹配的参数
# #         checkpoint = torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin"), map_location="cpu")
# #         model_dict = self.ner_model.state_dict()
# #
# #         # 过滤掉不匹配的键
# #         pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
# #         model_dict.update(pretrained_dict)
# #         self.ner_model.load_state_dict(model_dict, strict=False)
# #
# #         self.ner_model.to(self.device)
# #
# #     def ner_tokenizer(self, text):
# #         # 确保文本不超过最大长度（考虑 [CLS] 和 [SEP]）
# #         text = text[:self.max_seq_len - 2]  # 保留空间给 [CLS] 和 [SEP]
# #
# #         # 加上 [CLS] 和 [SEP] token
# #         text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
# #
# #         # 转换为 token IDs
# #         tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
# #
# #         # 填充 input_ids 和 attention_mask
# #         input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
# #         attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
# #
# #         # 转换为 torch tensors
# #         input_ids = torch.tensor([input_ids])
# #         attention_mask = torch.tensor([attention_mask])
# #
# #         return input_ids, attention_mask
# #
# #
# #     def ner_predict(self, text):
# #         input_ids, attention_mask = self.ner_tokenizer(text)
# #         input_ids = input_ids.to(self.device)
# #         attention_mask = attention_mask.to(self.device)
# #
# #         output = self.ner_model(input_ids, attention_mask)
# #
# #         attention_mask = attention_mask.detach().cpu().numpy()
# #         length = sum(attention_mask[0])
# #         logits = output.logits
# #
# #         if isinstance(logits, list):
# #             logits = torch.tensor(logits)
# #
# #         logits = logits[0][1:length - 1]
# #
# #         pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
# #
# #         pred_labels = [self.ner_id2label.get(int(i), "O") for i in pred_ids]
# #
# #         entities = get_entities(pred_labels)
# #
# #         result = {}
# #         for ent in entities:
# #             ent_name = ent[0]
# #             ent_start = ent[1]
# #             ent_end = ent[2]
# #             if ent_name not in result:
# #                 result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
# #             else:
# #                 result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
# #
# #         # 格式化为指定结构
# #         formatted_result = {
# #             "text": text,
# #             "trigger": [],
# #             "events": [],
# #             "time": [],
# #             "location": [],
# #             "cause": [],
# #             "loss": []
# #         }
# #
# #         label_mapping = {
# #             "Trigger": "trigger",
# #             "Event": "events",
# #             "Time": "time",
# #             "Location": "location",
# #             "Cause": "cause",
# #             "Loss": "loss"
# #         }
# #
# #         for label_type, spans in result.items():
# #             mapped_label = label_mapping.get(label_type)
# #             if mapped_label:
# #                 formatted_result[mapped_label] = [span[0] for span in spans]
# #
# #         return formatted_result
#
#
#     # def ner_predict(self, text):
#     #     input_ids, attention_mask = self.ner_tokenizer(text)
#     #     input_ids = input_ids.to(self.device)
#     #     attention_mask = attention_mask.to(self.device)
#     #
#     #     output = self.ner_model(input_ids, attention_mask)
#     #
#     #     attention_mask = attention_mask.detach().cpu().numpy()
#     #     length = sum(attention_mask[0])
#     #     logits = output.logits
#     #
#     #     # 如果 logits 是 list 类型，转换为 Tensor 类型
#     #     if isinstance(logits, list):
#     #         logits = torch.tensor(logits)
#     #
#     #     logits = logits[0][1:length - 1]
#     #
#     #     print(f"logits shape: {logits.shape}")  # 打印 logits 的形状
#     #     print(f"logits values: {logits}")  # 打印 logits 的值
#     #
#     #     logits = torch.tensor(logits)  # 确保 logits 是 Tensor 类型
#     #     pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()  # 获取预测的 ID
#     #     print("pred_ids:", pred_ids)  # 打印预测的 ID
#     #
#     #     # 处理 pred_ids 如果它是标量，转换为列表
#     #     if np.ndim(pred_ids) == 0:
#     #         pred_ids = [pred_ids]
#     #
#     #     pred_labels = [self.ner_id2label.get(int(i), "O") for i in pred_ids]
#     #     print("pred_labels:", pred_labels)  # 打印预测标签
#     #
#     #     # 查看实际标签及其位置
#     #     for idx, label in enumerate(pred_labels):
#     #         print(f"Token {input_ids[idx]} -> Label: {label}")
#     #
#     #     entities = get_entities(pred_labels)  # 提取实体
#     #     print("Extracted Entities:", entities)  # 打印提取的实体
#     #
#     #     result = {}
#     #     for ent in entities:
#     #         ent_name = ent[0]
#     #         ent_start = ent[1]
#     #         ent_end = ent[2]
#     #         if ent_name not in result:
#     #             result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
#     #         else:
#     #             result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
#     #
#     #     return result
#
#
# import torch
# import numpy as np
# from transformers import BertTokenizer
# from model import BertNer  # 替换为你自己的模型文件
#
#
# class Args:
#     def __init__(self, bert_dir, num_labels, max_seq_len):
#         self.bert_dir = bert_dir
#         self.num_labels = num_labels
#         self.max_seq_len = max_seq_len
#
#
# class Predictor:
#     def __init__(self, model_dir):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
#
#         self.label_list = ["O", "B-trigger", "I-trigger", "B-events", "I-events",
#                            "B-time", "I-time", "B-location", "I-location",
#                            "B-cause", "I-cause", "B-loss", "I-loss"]
#
#         self.id2label = {i: label for i, label in enumerate(self.label_list)}
#
#         # 创建 Args 对象，传递给 BertNer
#         args = Args(bert_dir="hfl/chinese-bert-wwm-ext", num_labels=len(self.label_list), max_seq_len=128)
#         self.args.num_labels = 13
#
#         self.model = BertNer(args)  # 传递 args
#
#         self.model.load_state_dict(torch.load(f"./checkpoint/{model_dir}/pytorch_model_ner.bin", map_location=self.device))
#         self.model.to(self.device)
#         self.model.eval()
#
#     def ner_predict(self, text):
#         encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding=False, add_special_tokens=True)
#         input_ids = encoding["input_ids"].to(self.device)
#         attention_mask = encoding["attention_mask"].to(self.device)
#
#         with torch.no_grad():
#             logits = self.model(input_ids, attention_mask=attention_mask)
#
#         pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
#
#         # 对 tokens 去掉 [CLS] 和 [SEP]
#         tokens = self.tokenizer.tokenize(text)
#         pred_ids = pred_ids[1:len(tokens) + 1]  # 注意加1是因为 [CLS] 的偏移
#         pred_labels = [self.id2label.get(int(i), "O") for i in pred_ids]
#
#         result = {}
#         current_label = None
#         current_text = ""
#
#         for token, label in zip(tokens, pred_labels):
#             if label.startswith("B-"):
#                 if current_label:
#                     result.setdefault(current_label, []).append((current_text,))
#                 current_label = label[2:]
#                 current_text = token.lstrip("##")
#             elif label.startswith("I-") and current_label == label[2:]:
#                 current_text += token.lstrip("##")
#             else:
#                 if current_label:
#                     result.setdefault(current_label, []).append((current_text,))
#                     current_label = None
#                     current_text = ""
#
#         if current_label and current_text:
#             result.setdefault(current_label, []).append((current_text,))
#
#         return result
#
#     def get_entities(pred_labels):
#         entities = []
#         entity = None
#         for i, label in enumerate(pred_labels):
#             if label.startswith("B-"):  # 开始新的实体
#                 if entity is not None:  # 处理前一个实体
#                     entities.append(entity)
#                 entity = [label[2:], i, i]  # 创建一个新实体，存储类型、起始位置、结束位置
#             elif label.startswith("I-") and entity is not None:  # 继续实体
#                 entity[2] = i  # 更新结束位置
#             else:  # 非实体
#                 if entity is not None:
#                     entities.append(entity)
#                 entity = None
#         if entity is not None:  # 处理最后一个实体
#             entities.append(entity)
#
#         return entities
#
#
#
# if __name__ == "__main__":
#     data_name = "dgre"
#     predictor = Predictor(data_name)
#     if data_name == "dgre":
#         texts = [
#             "山西太原一小区天然气管道泄漏，猛烈喷出数米高。2019年10月11日，太原迎新街一天然气管道泄漏，致使管道内的天然气猛烈喷涌而出。"
#             "2019年8月20日晚上，在浙江打工的小张，因为下意识的一个动作，身受重伤。据了解，当天小张下班后，突然想起忘记关掉仓库里的煤气罐，于是就赶回仓库查看。走到煤气罐的旁边，小张赶紧把阀门关了。但是由于仓库黑灯瞎火，小张为了检查煤气罐是否关紧，于是就掏出了“打火机”！！！"
#             "2019年10月18日凌晨0点左右，上海浦东归昌路凌河路交叉口的浦兴菜市场一面馆发生爆燃事故，据初步了解：不排除事发面馆内的液化气发生泄漏后发生爆燃事故，未造成人员伤亡。事故原因可能使用液化气操作不当。",
#         ]
#     elif data_name == "duie":
#         texts = [
#             "2024年6月13日，江苏南通。通州区川姜镇一沿街居民楼突然发生爆炸，通州区应急管理局工作人员称，事故原因可能是液化气瓶泄漏导致的燃爆，事故造成2人受伤，没有生命危险。",
#             "2024年4月10日6时15分左右，苏家屯区轻飏小筑小区一居民家中发生液化气爆燃事故，造成4人受伤，均已第一时间送医，其中1人抢救无效死亡。事故原因用户操作原因造成液化气泄漏爆燃。",
#
#         ]
#     for text in texts:
#         ner_result = predictor.ner_predict(text)
#         print("文本>>>>>：", text)
#         print("实体>>>>>：", ner_result)
#         print("="*100)
#
#
