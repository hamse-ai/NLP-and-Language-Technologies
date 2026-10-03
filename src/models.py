"""Neural architectures for approaches 3 and 4 (BiLSTM; CNN + BiGRU + Attention), kept small for CPU-only training."""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, n_classes=3, pad_idx=0,
                 dropout=0.4, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32),
                padding_idx=pad_idx, freeze=False,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # concat final forward/backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h))


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over GRU outputs."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out, mask):
        # gru_out: (B, T, H), mask: (B, T) with 1 for real tokens, 0 for padding
        scores = self.attn(gru_out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (gru_out * weights).sum(dim=1)  # (B, H)
        return context, weights.squeeze(-1)


class CNNBiGRUAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        n_filters=64,
        filter_sizes=(2, 3, 4),
        gru_hidden=96,
        n_classes=3,
        pad_idx=0,
        dropout=0.4,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in filter_sizes]
        )
        cnn_out_dim = n_filters * len(filter_sizes)
        self.gru = nn.GRU(cnn_out_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.attention = Attention(gru_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, n_classes)

    def forward(self, x):
        mask = (x != self.pad_idx).float()
        emb = self.embedding(x).transpose(1, 2)  # (B, E, T)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]  # each (B, F, T')
        min_len = min(c.size(2) for c in conv_outs)
        conv_outs = [c[:, :, :min_len] for c in conv_outs]
        cnn_feat = torch.cat(conv_outs, dim=1).transpose(1, 2)  # (B, T', F*len)
        mask_t = mask[:, :min_len]
        gru_out, _ = self.gru(cnn_feat)
        context, _ = self.attention(gru_out, mask_t)
        return self.fc(self.dropout(context))
