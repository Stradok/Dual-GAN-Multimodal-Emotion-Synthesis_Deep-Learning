import torch
import torch.nn as nn

class CaptionTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, nhead=8, num_layers=6, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, tgt_seq, memory):
        """
        tgt_seq: [seq_len, batch] -> token indices
        memory: [batch, emb_dim] -> context vector
        """
        tgt = self.embedding(tgt_seq) + self.positional_encoding[:, :tgt_seq.size(0), :]
        memory = memory.unsqueeze(0)  # [1, batch, emb_dim]
        output = self.transformer_decoder(tgt, memory)
        return self.fc(output)
