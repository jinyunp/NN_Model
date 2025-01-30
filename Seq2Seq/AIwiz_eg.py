# -*- coding: cp949 -*-
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import MarianTokenizer

# 학습용 데이터셋 구조체
class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data.iloc[idx, 0], self.data.iloc[idx, 1]

# Multi-Head Attention 계층 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model})은 n_heads ({n_heads})로 나누어 떨어져야 합니다.'

        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_score = Q @ K.permute(0, 1, 3, 2) / self.scale

        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -1e10)

        attention_dist = torch.softmax(attention_score, dim=-1)
        attention = attention_dist @ V

        x = attention.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        x = self.fc_o(x)

        return x, attention_dist

# 순전파 구현 클래스
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.linear(x)
        return x

# Encoder 계층 클래스
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()

        self.self_atten = MultiHeadAttention(d_model, n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):
        x_norm = self.LN(x)
        output, atten_enc = self.self_atten(x_norm, x_norm, x_norm, enc_mask)
        x = x + self.dropout(output)
        x_norm = self.LN(x)
        output = self.FF(x_norm)
        x = x_norm + self.dropout(output)

        return x, atten_enc

# Encoder 블록 클래스
class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)
        self.LN = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])

    def forward(self, src, mask, atten_map_save=False):
        pos = torch.arange(src.shape[1], device=src.device).repeat(src.shape[0], 1)

        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_encs = []
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save:
                atten_encs.append(atten_enc[0].unsqueeze(0))

        if atten_map_save:
            atten_encs = torch.cat(atten_encs, dim=0)

        x = self.LN(x)
        return x, atten_encs

# Decoder 계층 클래스
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.atten = MultiHeadAttention(d_model, n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        x, atten_dec = self.process_sublayer(x, self.atten, self.LN, dec_mask)
        x, atten_enc_dec = self.process_sublayer(x, self.atten, self.LN, enc_dec_mask, enc_out)
        x, _ = self.process_sublayer(x, self.FF, self.LN)

        return x, atten_dec, atten_enc_dec

    def process_sublayer(self, x, sublayer, norm_layer, mask=None, enc_out=None):
        x_norm = norm_layer(x)
        if isinstance(sublayer, MultiHeadAttention):
            if enc_out is not None:
                residual, atten = sublayer(x_norm, enc_out, enc_out, mask)
            else: 
                residual, atten = sublayer(x_norm, x_norm, x_norm, mask)
        elif isinstance(sublayer, FeedForward):
            residual = sublayer(x_norm)
            atten = None 
        else:
            raise TypeError("Unsupported sublayer type")

        return x + self.dropout(residual), atten

# Decoder 블록 클래스
class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size):

        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])
        self.LN = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save=False):

        pos = torch.arange(trg.shape[1], device=trg.device).repeat(trg.shape[0], 1)

        x = self.scale * self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_decs = []
        atten_enc_decs = []
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save:
                atten_decs.append(atten_dec[0].unsqueeze(0))
                atten_enc_decs.append(atten_enc_dec[0].unsqueeze(0))

        if atten_map_save:
            atten_decs = torch.cat(atten_decs, dim=0)
            atten_enc_decs = torch.cat(atten_enc_decs, dim=0)

        x = self.LN(x)
        x = self.fc_out(x)

        return x, atten_decs, atten_enc_decs

# 모델 정의 클래스
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx):

        super().__init__()
        self.pad_idx = pad_idx
        input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p)
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size)

        self.n_heads = n_heads

        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)

    def make_enc_mask(self, src):
        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_mask.repeat(1, self.n_heads, src.shape[1], 1).to(src.device)

    def make_dec_mask(self, trg):
        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(trg.device)
        trg_dec_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1], device=trg.device))==0
        dec_mask = trg_pad_mask | trg_dec_mask
        return dec_mask

    def make_enc_dec_mask(self, src, trg):
        enc_dec_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(src.device)

    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)

        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)

        return out, atten_encs, atten_decs, atten_enc_decs

# 추론용 데이터셋 구조체
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# 번역 함수
def batch_translation(model, src_texts, tokenizer, extra_token_length=50):
    translated_texts = []
    
    with torch.no_grad():
        src = tokenizer(src_texts, padding=True, truncation=True, return_tensors='pt')
        enc_mask = model.make_enc_mask(src.input_ids)
        enc_out, _ = model.encoder(src.input_ids, enc_mask)
        
        max_output_length = src.input_ids.shape[1] + extra_token_length

        bos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        pred = torch.full((len(src_texts), 1), bos_token_id, dtype=torch.long)
        
        for _ in tqdm(range(max_output_length),desc='Batch Translating',unit='sentences',leave=False):
            dec_mask = model.make_dec_mask(pred)
            enc_dec_mask = model.make_enc_dec_mask(src.input_ids, pred)
            out, _, _ = model.decoder(pred, enc_out, dec_mask, enc_dec_mask)

            pred_word = out.argmax(dim=2)[:, -1].unsqueeze(1)
            pred = torch.cat([pred, pred_word], dim=1)
            
            if (pred_word == tokenizer.eos_token_id).all():
                break

        for p in pred:
            translated_text = tokenizer.decode(p, skip_special_tokens=True)
            translated_texts.append(translated_text)

    return translated_texts

# 모델 학습 함수
def Train(model,train_DL,criterion,optimizer,params):

    BATCH_SIZE = params['batch_size']
    EPOCH = params['epoch']
    max_len = params['max_len']
    tokenizer = params['tokenizer']
    earlyStopping = params['early_stop']

    history = {"train": [], "val": [], "lr":[]}
    best_loss = float('inf')

    train_losses = []
    for ep in range(EPOCH):
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, optimizer=optimizer, max_len=max_len, tokenizer=tokenizer, epoch = ep+1, total_epochs=EPOCH)
        history["train"].append(train_loss)

        current_lr = optimizer.param_groups[0]['lr'] 
        history["lr"].append(current_lr)

        print(f'Epoch [{ep + 1}/{EPOCH}], Loss: {train_loss:.4f}', flush=True)

        train_losses.append(train_loss)

        earlyStopping(train_loss, model)
        if earlyStopping.early_stop:
            break

    return train_losses

# 모델 학습 세부 함수
def loss_epoch(model, DL, criterion, optimizer=None, max_len=None, DEVICE=None, tokenizer=None, scheduler=None, epoch=None, total_epochs=None):
    N = len(DL.dataset)

    rloss = 0
    tqdm_desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else None

    for src_texts, trg_texts in tqdm(DL, desc=tqdm_desc, unit="batch"):
        src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = tokenizer(trg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
        
        y_hat = model(src, trg[:, :-1])[0] 
        loss = criterion(y_hat.permute(0, 2, 1), trg[:, 1:]) 
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_b = loss.item() * src.shape[0]
        rloss += loss_b

    loss_e = rloss / N
    return loss_e

# EarlyStop 구현 클래스
class EarlyStopping:
        def __init__(self, patience=10, verbose=False, delta=0, target_loss=None):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.target_loss = target_loss

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.target_loss is not None and val_loss <= self.target_loss:
                self.early_stop = True
                if self.verbose:
                    print(f'Target loss {self.target_loss} reached. Stopping training.',flush=True)
                return

            if self.best_score is None:
                self.best_score = score
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose :
                        print(f'No improvement for {self.patience} epochs. Early stopping.',flush=True)
            else:
                self.best_score = score
                self.counter = 0

def main():
    # ==================================================
    # Step 1: 데이터 준비
    # Pandas DataFrame 형식으로 학습 데이터를 준비합니다.
    # 학습 데이터는 ecmData 변수에 저장되어야 합니다. 
    # 입력 데이터 예시
    ecmData = pd.read_csv("학습 데이터 경로 입력", encoding='cp949')
    # ==================================================
    # Step 2: 데이터 전처리
    custom_DS = CustomDataset(ecmData)
    train_DL = torch.utils.data.DataLoader(custom_DS, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TranslationDataset(ecmData.iloc[:,0])  
    test_DL = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    # ==================================================
    # Step 3: 모델 및 학습 설정
    BATCH_SIZE = 64
    EPOCH = 100
    max_len = 512
    drop_p = 0.1

    d_model = 128
    n_heads = 4
    n_layers = 3

    d_ff = 512 
    LR_scale = 1
    label_smoothing=float(0.1)
    criterion_type = 'ce'

    tokenizer = MarianTokenizer.from_pretrained(r"C:\Program Files\ECMiner\ECMiner_x64_withVOD_v.5.2.0.7592_20240719\Miniconda3\ModelSrc\Transformer_Transformer\model_directory")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id

    params = {}
    params['batch_size'] = int(BATCH_SIZE)
    params['epoch'] = int(EPOCH) 
    params['max_len'] = max_len
    params['tokenizer'] = tokenizer
    params['early_stop'] = EarlyStopping(verbose=True, target_loss = 0.0001)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    model = Transformer(vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx)
    optimizer = optim.Adam(model.parameters()
                           , lr = 0.001
                           , betas = (0.9, 0.98)
                           , eps = 1e-9
                           , weight_decay=1e-5
                           )
    # ==================================================
    # Step 4: 모델 학습
    train_losses = Train(model, train_DL, criterion, optimizer, params)
    # ==================================================
    # Step 5: 학습 결과 생성
    model.eval()

    all_translations = []
    total_batches = len(test_DL)
    
    print(f'결과를 생성중입니다...', flush=True)
    with tqdm(total=total_batches, desc="Total", unit='batch') as pbar:
        for batch in test_DL : 
            translations = batch_translation(model, batch, tokenizer)
            all_translations.extend(translations)
            pbar.update(1)

    ecmData['YHAT'] = all_translations
    ecmData = ecmData['YHAT'].replace('', '!!!Translation failed due to insufficient model training!!!')
    # ==================================================
    # Step 6: 결과 및 모델 저장
    torch.save(model, "모델 저장 경로 입력")
    ecmData.to_csv("결과 저장 경로 입력", index = False, encoding = 'CP949')

if __name__ == "__main__":
    main()