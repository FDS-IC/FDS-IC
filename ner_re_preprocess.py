# -*- coding: utf-8 -*-
import os
import json
import random
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataPreprocessor:
    def __init__(self, data_path="./data/dgre"):
        self.data_path = data_path
        self.rels = set()
        self.labels = set()

    def ensure_dir(self, dir_name):
        full_path = os.path.join(self.data_path, dir_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def split_data(self, data_list, train_file, dev_file):
        train_data, dev_data = train_test_split(data_list, test_size=0.1, random_state=42)
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(dev_file, "w", encoding="utf-8") as f:
            for item in dev_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

class ProcessDgreData(DataPreprocessor):
    def get_ner_data(self):
        with open(os.path.join(self.data_path, "train.json"), encoding="utf-8") as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        result = []
        for item in raw_data:
            text = item["text"]
            labels = ["O"] * len(text)
            for spo in item["spo_list"]:
                for ent, tag in [(spo["subject"], "B-故障设备"), (spo["object"], "B-故障原因")]:
                    start_idx = text.find(ent)
                    if start_idx != -1:
                        labels[start_idx] = tag
                        for i in range(1, len(ent)):
                            if start_idx + i < len(text):
                                labels[start_idx + i] = tag.replace("B-", "I-")

            result.append({"text": list(text), "labels": labels})

        self.labels = sorted(set(l for r in result for l in r["labels"] if l != "O")) + ["O"]
        ner_dir = self.ensure_dir("ner_data")
        self.split_data(result, os.path.join(ner_dir, "train.txt"), os.path.join(ner_dir, "dev.txt"))

        with open(os.path.join(ner_dir, "labels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.labels))

    def get_re_data(self):
        with open(os.path.join(self.data_path, "train.json"), encoding="utf-8") as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        result = []
        for item in raw_data:
            text = item["text"]
            spo_list = item["spo_list"]
            pos_rels = [[spo["subject"], spo["object"], spo["predicate"]] for spo in spo_list]
            neg_rels = []
            subjects = list(set(spo["subject"] for spo in spo_list))
            objects = list(set(spo["object"] for spo in spo_list))
            for subj in subjects:
                for obj in objects:
                    if [subj, obj] not in [[r[0], r[1]] for r in pos_rels]:
                        neg_rels.append([subj, obj, "没关系"])

            for triple in pos_rels + neg_rels:
                result.append({"text": text, "triple": triple})
                self.rels.add(triple[2])

        re_dir = self.ensure_dir("re_data")
        self.split_data(result, os.path.join(re_dir, "train.txt"), os.path.join(re_dir, "dev.txt"))

        with open(os.path.join(re_dir, "rels.txt"), "w", encoding="utf-8") as f:
            json.dump(sorted(self.rels), f, ensure_ascii=False, indent=2)

class ProcessDuieData(DataPreprocessor):
    def get_ents(self):
        schema_file = os.path.join(self.data_path, "duie_schema.json")
        with open(schema_file, encoding="utf-8") as f:
            schema = json.load(f)

        rel_dict = defaultdict(set)
        for item in schema:
            rel_dict[item["subject_type"] + "#" + item["object_type"]].add(item["predicate"])
            self.labels.update(["B-" + item["subject_type"], "I-" + item["subject_type"],
                                "B-" + item["object_type"], "I-" + item["object_type"]])

        ner_dir = self.ensure_dir("ner_data")
        re_dir = self.ensure_dir("re_data")

        with open(os.path.join(ner_dir, "labels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(self.labels)))
        with open(os.path.join(re_dir, "rels.txt"), "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in rel_dict.items()}, f, ensure_ascii=False, indent=2)

    def get_ner_data(self, input_file, output_file):
        with open(input_file, encoding="utf-8") as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        result = []
        for item in raw_data:
            text = item["text"]
            labels = ["O"] * len(text)
            for spo in item.get("spo_list", []):
                for ent, tag in [(spo["subject"], "B-" + spo["subject_type"]),
                                 (spo["object"], "B-" + spo["object_type"])]:
                    for match in re.finditer(re.escape(ent), text):
                        start = match.start()
                        labels[start] = tag
                        for i in range(1, len(ent)):
                            if start + i < len(text):
                                labels[start + i] = tag.replace("B-", "I-")

            result.append({"text": list(text), "labels": labels})

        with open(output_file, "w", encoding="utf-8") as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    processor = ProcessDgreData("./data/dgre")
    processor.get_ner_data()
    processor.get_re_data()

    duie_processor = ProcessDuieData("./data/duie")
    duie_processor.get_ents()
    duie_processor.get_ner_data("./data/duie/train.json", "./data/duie/ner_data/train.txt")
    duie_processor.get_ner_data("./data/duie/dev.json", "./data/duie/ner_data/dev.txt")
