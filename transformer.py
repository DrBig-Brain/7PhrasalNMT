import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    def __init__(self, vocab_size, sequence_len, nblock, nhead, embed_dim, phrase_emb_dim):
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.nblock = nblock
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.phrase_emb_dim = phrase_emb_dim

class TransformerWithPhrase(nn.Module):
    def __init__(self, config: TransformerConfig, phrase_vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim - config.phrase_emb_dim)
        self.phrase_embedding = nn.Embedding(phrase_vocab_size, config.phrase_emb_dim)
        self.pos_embedding = nn.Embedding(config.sequence_len, config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.nblock)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim, nhead=config.nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.nblock)
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt, src_phrase, tgt_phrase):
        bs, src_len = src.shape
        bs, tgt_len = tgt.shape
        src_tok_emb = self.token_embedding(src)
        src_phrase_emb = self.phrase_embedding(src_phrase)
        src_emb = torch.cat([src_tok_emb, src_phrase_emb], dim=-1)
        src_emb = src_emb + self.pos_embedding(torch.arange(src_len, device=src.device))[None]
        memory = self.encoder(src_emb)

        tgt_tok_emb = self.token_embedding(tgt)
        tgt_phrase_emb = self.phrase_embedding(tgt_phrase)
        tgt_emb = torch.cat([tgt_tok_emb, tgt_phrase_emb], dim=-1)
        tgt_emb = tgt_emb + self.pos_embedding(torch.arange(tgt_len, device=tgt.device))[None]
        output = self.decoder(tgt_emb, memory)
        return self.fc_out(output)

    def generate(self, src, src_phrase, max_len=30, start_token=1):
        self.eval()
        with torch.no_grad():
            bs = src.shape[0]
            src_tok_emb = self.token_embedding(src)
            src_phrase_emb = self.phrase_embedding(src_phrase)
            src_emb = torch.cat([src_tok_emb, src_phrase_emb], dim=-1)
            src_emb = src_emb + self.pos_embedding(torch.arange(src.shape[1], device=src.device))[None]
            memory = self.encoder(src_emb)
            tgt = torch.full((bs, 1), start_token, dtype=torch.long, device=src.device)
            tgt_phrase = torch.zeros_like(tgt)
            outputs = []
            for _ in range(max_len):
                tgt_tok_emb = self.token_embedding(tgt)
                tgt_phrase_emb = self.phrase_embedding(tgt_phrase)
                tgt_emb = torch.cat([tgt_tok_emb, tgt_phrase_emb], dim=-1)
                tgt_emb = tgt_emb + self.pos_embedding(torch.arange(tgt.shape[1], device=src.device))[None]
                out = self.decoder(tgt_emb, memory)
                next_token = self.fc_out(out[:, -1, :]).argmax(-1, keepdim=True)
                outputs.append(next_token)
                tgt = torch.cat([tgt, next_token], dim=1)
                tgt_phrase = torch.cat([tgt_phrase, torch.zeros_like(next_token)], dim=1)
            return torch.cat(outputs, dim=1)
