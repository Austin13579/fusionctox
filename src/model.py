import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_dim=128, dim=128, output_dim=256):
        super(FiLM, self).__init__()
        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

    def forward(self, x, y):
        gamma, beta = torch.split(self.fc(x), self.dim, 1)
        output = gamma * y + beta
        return self.fc_out(output)


class Char_Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(Char_Encoder, self).__init__()
        self.char_embedding = nn.Embedding(64, embed_dim)

        self.char_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),

            nn.BatchNorm1d(embed_dim),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.BatchNorm1d(embed_dim),
        )
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seq):
        a_emb = self.char_embedding(seq)
        a_emb = a_emb.transpose(2, 1)
        a_emb = self.char_conv(a_emb)
        a_emb1 = self.max_pool(a_emb).squeeze()
        a_emb2 = self.avg_pool(a_emb).squeeze()
        return a_emb1+a_emb2


class BPE_Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(BPE_Encoder, self).__init__()
        self.sub_embedding = nn.Embedding(498, embed_dim)
        self.sub_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.BatchNorm1d(embed_dim),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.BatchNorm1d(embed_dim),
        )
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seq):
        s_emb = self.sub_embedding(seq)
        s_emb = s_emb.transpose(2, 1)
        s_emb = self.sub_conv(s_emb)
        s_emb1 = self.max_pool(s_emb).squeeze()
        s_emb2 = self.avg_pool(s_emb).squeeze()
        return s_emb1+s_emb2

class FP_Encoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(FP_Encoder, self).__init__()
        self.fp_encoder = nn.Sequential(
            nn.Linear(1024 + 881, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim)
        )

    def forward(self, fp):
        fp_emb = self.fp_encoder(fp)
        return fp_emb

class Decoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim,256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
    def forward(self,x):
        return x


class Char_Model(nn.Module):
    def __init__(self):
        super(Char_Model, self).__init__()
        # Encoder for Character-level SMILES
        self.char_encoder=Char_Encoder()
        # Decoder
        self.decoder = Decoder()

    def forward(self, seq):
        rep=self.char_encoder(seq)
        out=self.decoder(rep)
        return out


class BPE_Model(nn.Module):
    def __init__(self):
        super(BPE_Model, self).__init__()
        # Encoder for BPE-level SMILES
        self.bpe_encoder=BPE_Encoder()
        # Decoder
        self.decoder = Decoder()

    def forward(self, seq):
        rep=self.bpe_encoder(seq)
        out=self.decoder(rep)
        return out


class FP_Model(nn.Module):
    def __init__(self):
        super(FP_Model, self).__init__()

        # Encoder for Molecular Image
        self.fp_encoder=FP_Encoder()

        # Decoder
        self.decoder = Decoder()

    def forward(self, fp):
        rep=self.fp_encoder(fp)
        out=self.decoder(rep)
        return out


class FusionCTox_Sum(nn.Module):
    def __init__(self):
        super(FusionCTox_Sum, self).__init__()

        ### Moleculer Encoders
        self.fp_encoder = FP_Encoder()  # Encoder for Fingerprint
        # Encoder for Molecular Sequence
        self.char_encoder=Char_Encoder()
        self.bpe_encoder = BPE_Encoder()
        ### Decoders
        self.decoder=Decoder(128*1)

    def forward(self,fp,seq1,seq2):
        # ECFP Embedding
        fp_emb=self.fp_encoder(fp)

        # Molecular Sequence Embedding
        a_emb=self.char_encoder(seq1)
        s_emb=self.bpe_encoder(seq2)
        rep=fp_emb+a_emb+s_emb
        out=self.decoder(rep)
        return out


class FusionCTox_Film(nn.Module):
    def __init__(self):
        super(FusionCTox_Film, self).__init__()

        ### Moleculer Encoders
        self.fp_encoder = FP_Encoder()  # Encoder for Fingerprint
        # Encoder for Molecular Sequence
        self.char_encoder=Char_Encoder()
        self.bpe_encoder = BPE_Encoder()
        self.film=FiLM()
        ### Decoders
        self.decoder=Decoder(128*2)

    def forward(self,fp,seq1,seq2):
        # ECFP Embedding
        fp_emb=self.fp_encoder(fp)

        # Molecular Sequence Embedding
        a_emb=self.char_encoder(seq1)
        s_emb=self.bpe_encoder(seq2)
        rep=self.film(fp_emb,a_emb+s_emb)

        out=self.decoder(rep)
        return out

class FusionCTox_Concat(nn.Module):
    def __init__(self):
        super(FusionCTox_Concat, self).__init__()

        ### Moleculer Encoders
        self.fp_encoder = FP_Encoder()  # Encoder for Fingerprint
        # Encoder for Molecular Sequence
        self.char_encoder=Char_Encoder()
        self.bpe_encoder = BPE_Encoder()
        ### Decoders
        self.decoder=Decoder(128*3)

    def forward(self,fp,seq1,seq2):
        # ECFP Embedding
        fp_emb=self.fp_encoder(fp)

        # Molecular Sequence Embedding
        a_emb=self.char_encoder(seq1)
        s_emb=self.bpe_encoder(seq2)
        rep=torch.cat((fp_emb, a_emb,s_emb), dim=1)
        out=self.decoder(rep)
        return out