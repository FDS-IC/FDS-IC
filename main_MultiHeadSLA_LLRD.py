import os
import json
import torch
import numpy as np
from tqdm import tqdm
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel
from torch.optim import AdamW
from config_SLA_LLRD import NerConfig
from model_MultiHeadSLA_LLRD import BertNerWithAttention_SLA_LLRD
from data_loader import NerDataset


class Trainer:
    def __init__(self, output_dir=None, model=None, train_loader=None, dev_loader=None,
                 test_loader=None, optimizer=None, schedule=None, epochs=1,
                 device="cpu", id2label=None, save_step=500):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.schedule = schedule
        self.epochs = epochs
        self.device = device
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(train_loader) * epochs

    def train(self):
        self.model.to(self.device)
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for step, batch_data in enumerate(self.train_loader):
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(self.device)

                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]

                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.schedule:
                    self.schedule.step()

                print(f"[Train] Epoch {epoch}/{self.epochs} - Step {global_step}/{self.total_step} - Loss: {loss.item():.4f}")
                global_step += 1

                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_MultiHeadnerLLRD.bin"))

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_MultiHeadnerLLRD.bin"))

    def test(self):
        self.model.to(self.device)
        checkpoint_path = os.path.join(self.output_dir, "pytorch_model_MultiHeadnerLLRD.bin")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_data in tqdm(self.test_loader, desc="Testing"):
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(self.device)

                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]

                output = self.model(input_ids, attention_mask, labels)
                logits = output.logits
                pred_ids = logits

                label_ids = labels.detach().cpu().numpy()
                mask = attention_mask.detach().cpu().numpy()

                for i in range(input_ids.size(0)):
                    length = int(mask[i].sum())
                    pred_seq = [self.id2label.get(int(idx), "O") for idx in pred_ids[i][:length][1:]]
                    true_seq = [self.id2label.get(int(idx), "O") for idx in label_ids[i][:length][1:]]
                    preds.append(pred_seq)
                    trues.append(true_seq)

        report_dict = classification_report(trues, preds, output_dict=True, zero_division=1)
        output_keys = ['micro avg', 'macro avg', 'weighted avg']
        print(f"{'':<15}{'precision':<10}{'recall':<10}{'f1-score':<10}")
        for key in output_keys:
            print(f"{key:<15}{report_dict[key]['precision']:<10.2f}{report_dict[key]['recall']:<10.2f}{report_dict[key]['f1-score']:<10.2f}")
        return report_dict


# 👇 三层分层学习率 + 模块分组优化器
def build_optimizer_and_scheduler(args, model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    module = model.module if hasattr(model, "module") else model

    bert_params = list(module.bert.named_parameters())

    bottom_layer = [p for n, p in bert_params if any(f"encoder.layer.{i}." in n for i in [0, 1, 2])]
    middle_layer = [p for n, p in bert_params if any(f"encoder.layer.{i}." in n for i in [3, 4, 5, 6, 7])]
    top_layer = [p for n, p in bert_params if any(f"encoder.layer.{i}." in n for i in [8, 9, 10, 11])]
    other_bert = [p for n, p in bert_params if "encoder.layer" not in n]

    lstm_params = [p for n, p in module.named_parameters() if "bilstm" in n]
    linear_params = [p for n, p in module.named_parameters() if "linear" in n or "classifier" in n]
    crf_params = [p for n, p in module.named_parameters() if "crf" in n]
    sla_params = [p for n, p in module.named_parameters() if "sla" in n]

    grouped_parameters = [
        {"params": bottom_layer, "lr": args.bert_learning_rate * 0.5, "weight_decay": args.weight_decay},
        {"params": middle_layer, "lr": args.bert_learning_rate * 0.75, "weight_decay": args.weight_decay},
        {"params": top_layer + other_bert, "lr": args.bert_learning_rate, "weight_decay": args.weight_decay},
        {"params": lstm_params, "lr": args.bilstm_learning_rate, "weight_decay": args.weight_decay},
        {"params": linear_params, "lr": args.linear_learning_rate, "weight_decay": args.weight_decay},
        {"params": crf_params, "lr": args.crf_learning_rate, "weight_decay": args.weight_decay},
        {"params": sla_params, "lr": args.sla_learning_rate, "weight_decay": args.weight_decay},
    ]

    optimizer = AdamW(grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_ratio),
        num_training_steps=t_total
    )
    return optimizer, scheduler


def main(data_name):
    args = NerConfig(data_name)

    with open(os.path.join(args.data_path, "labels_1.txt"), "r", encoding="utf-8") as fp:
        labels = fp.read().splitlines()
    args.label2id = {label: idx for idx, label in enumerate(labels)}
    args.id2label = {idx: label for label, idx in args.label2id.items()}

    with open(os.path.join(args.output_dir, "ner_args.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(args.data_path, "train_1.txt"), "r", encoding="utf-8") as fp:
        train_data = [json.loads(d) for d in fp.read().splitlines() if d.strip()]
    with open(os.path.join(args.data_path, "dev_1.txt"), "r", encoding="utf-8") as fp:
        dev_data = [json.loads(d) for d in fp.read().splitlines() if d.strip()]

    train_loader = DataLoader(NerDataset(train_data, args, tokenizer), batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(NerDataset(dev_data, args, tokenizer), batch_size=args.dev_batch_size, shuffle=False)

    model = BertNerWithAttention_SLA_LLRD(args).to(device)
    t_total = len(train_loader) * args.epochs

    optimizer, scheduler = build_optimizer_and_scheduler(args, model, t_total)

    trainer = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=scheduler,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label
    )

    trainer.train()
    report = trainer.test()
    print(report)


if __name__ == "__main__":
    data_name = "dgre"
    main(data_name)
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
