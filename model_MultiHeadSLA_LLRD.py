import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig


class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


# 多头 Self-Local Attention 模块
class SelfLocalAttention(nn.Module):
    def __init__(self, hidden_size, window_size, num_heads=8):
        super(SelfLocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, value, key, query, mask=None):
        batch_size, seq_len, _ = query.size()
        window_size = self.window_size

        Q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]

        # 构造本地窗口掩码
        local_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=query.device)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            local_mask[i, start:end] = 1
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        scores = scores.masked_fill(local_mask == 0, -1e9)

        weights = self.softmax(scores)  # [B, H, L, L]
        out = torch.matmul(weights, V)  # [B, H, L, D]

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [B, L, H*D]
        return self.fc_out(out)  # [B, L, hidden_size]


# BERT + BiLSTM + 多头SLA + Linear + CRF 模型
class BertNerWithAttention_SLA_LLRD(nn.Module):
    def __init__(self, args):
        super(BertNerWithAttention_SLA_LLRD, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        hidden_size = self.bert_config.hidden_size

        self.lstm_hidden = 64
        self.max_seq_len = args.max_seq_len

        self.bilstm = nn.LSTM(
            hidden_size,
            self.lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # 使用多头 SLA，head 数设为 8（可调）
        self.sla = SelfLocalAttention(self.lstm_hidden * 2, window_size=5, num_heads=8)

        # 为融合后的输出加一个线性变换层，确保维度一致
        self.fc_sla = nn.Linear(self.lstm_hidden * 2, self.lstm_hidden * 2)  # [B, L, 2*H]

        self.layernorm = nn.LayerNorm(self.lstm_hidden * 2)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.lstm_hidden * 2, args.num_labels)
        self.crf = CRF(args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Step 1: BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [B, L, H]

        # Step 2: BiLSTM
        lstm_out, _ = self.bilstm(seq_out)
        lstm_out = self.dropout(lstm_out)  # [B, L, 2*H]

        # Step 3: SLA + 残差连接
        sla_out = self.sla(lstm_out, lstm_out, lstm_out, mask=attention_mask)  # [B, L, 2*H]

        # 投影 SLA 输出并加入残差连接
        sla_out = self.fc_sla(sla_out)  # [B, L, 2*H]
        fused_out = lstm_out + sla_out  # [B, L, 2*H]
        fused_out = self.layernorm(fused_out)  # [B, L, 2*H]

        # Step 4: Linear
        fused_out = self.dropout(fused_out)
        logits = self.linear(fused_out)  # [B, L, num_labels]

        # Step 5: CRF
        pred = self.crf.decode(logits, mask=attention_mask.bool())
        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')

        return ModelOutput(pred, labels, loss)
