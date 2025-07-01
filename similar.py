import os
import pandas as pd
from typing import List

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
class InputExample:
    """A single training/test example for sequence classification."""

    def __init__(self, guid: str, text_a: str, text_b: str = None, label: str = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor:
    """Base processor for sequence classification datasets."""

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        raise NotImplementedError()

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        raise NotImplementedError()

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        raise NotImplementedError()


class SimProcessor(DataProcessor):
    """Processor for similarity classification (e.g., sentence pairs)."""

    def _create_examples(self, df: pd.DataFrame, set_type: str) -> List[InputExample]:
        examples = []
        for i, row in df.iterrows():
            guid = f"{set_type}-{i}"
            text_a = str(row[0])
            text_b = str(row[1])
            label = str(row[2]) if len(row) > 2 else None
            examples.append(InputExample(guid, text_a, text_b, label))
        return examples

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(path)
        return self._create_examples(df, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        path = os.path.join(data_dir, "dev.csv")
        df = pd.read_csv(path)
        return self._create_examples(df, "dev")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        path = os.path.join(data_dir, "test.csv")
        df = pd.read_csv(path)
        return self._create_examples(df, "test")

    def get_sentence_examples(self, questions: List[List[str]]) -> List[InputExample]:
        examples = []
        for i, pair in enumerate(questions):
            guid = f"predict-{i}"
            text_a = str(pair[0])
            text_b = str(pair[1])
            label = "0"  # default label
            examples.append(InputExample(guid, text_a, text_b, label))
        return examples

    def get_labels(self) -> List[str]:
        return ["0", "1"]


class BertSimModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', num_labels=2):
        super(BertSimModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertSim:
    def __init__(self, model_name='bert-base-chinese', batch_size=32, max_seq_length=128, num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertSimModel(model_name, num_labels)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def encode_examples(self, examples):
        """Convert InputExample list to features tensor."""
        inputs = self.tokenizer(
            [(e.text_a, e.text_b) for e in examples],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        labels = torch.tensor([int(e.label) for e in examples], dtype=torch.long)
        return inputs, labels

    def train(self, train_dataset, learning_rate=2e-5, epochs=3):
        self.model.train()
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                optimizer.zero_grad()
                logits = self.model(**inputs)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def evaluate(self, eval_dataset):
        self.model.eval()
        dataloader = DataLoader(eval_dataset, batch_size=self.batch_size)
        preds, targets = [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                logits = self.model(**inputs)
                predictions = torch.argmax(logits, dim=-1)
                preds.extend(predictions.cpu().tolist())
                targets.extend(labels.cpu().tolist())

        acc = accuracy_score(targets, preds)
        auc = roc_auc_score(targets, preds)
        print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return acc, auc
