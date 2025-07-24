# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import MarianTokenizer


# 학습용 데이터셋 구조체
class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        src = self.data.iloc[idx, 0]
        trg = self.data.iloc[idx, 1]
        return (src, trg)

# 추론용 데이터셋 구조체
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Encoder class using LSTM
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, drop_p):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_p, batch_first=True)
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell
    
# Decoder class using LSTM
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, drop_p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_p, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
    
# Seq2Seq class combining Encoder and Decoder
class Seq2Seq(nn.Module):   ##########################################################
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, params, src, trg=None, trg_vocab_size=None, teacher_forcing_ratio=0.5, inference=False):
        max_len = params['max_len']
        batch_size = src.shape[0]
        if trg is not None:
            trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, max_len if inference else trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        hidden, cell = self.encoder(src)
        
        # If we are in inference mode, we don't use the actual target sequence
        if inference:
            # We assume that <SOS> (start-of-sequence) token is the first token (index 0)
            x = torch.zeros(batch_size).long().to(self.device)  # SOS token
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                outputs[:, t] = output
                top1 = output.argmax(1)  # Get the token with the highest probability
                x = top1  # Set the next input as the predicted token
                # You could also add an end-of-sequence (EOS) check here to stop early
        else:
            # Use the target sequence during training
            x = trg[:, 0]  # First input to the decoder is the <SOS> token (start token)
            for t in range(1, trg_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                outputs[:, t] = output
                top1 = output.argmax(1)
                # Decide whether to use teacher forcing
                x = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
        
        return outputs

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

def loss_epoch(model, dataloader, criterion, optimizer, params, epoch):
    epoch_loss = 0
    tokenizer=params['tokenizer']
    max_len = params['max_len']

    for batch_idx, (src, trg) in enumerate(dataloader):
        print(f'{batch_idx+1}/{len(dataloader)} batch processing within #{epoch+1} epoch', end='\r')
        # Move both src and trg to the device
        src = tokenizer(src, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
        # trg_texts = ['</s> ' + s for s in trg]
        trg = tokenizer(trg, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
        
        optimizer.zero_grad()
        output = model(params, src, trg, trg_vocab_size=params['tokenizer'].vocab_size)   #############################
        # output = model(src, trg, trg_vocab_size=params['tokenizer'].vocab_size)
        
        # Reshape output and target for loss computation
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)

        # loss = criterion(output.permute(0, 2, 1), trg[:, 1:]) 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def Train(model, dataloader, criterion, optimizer, params):
    model.train()
    history = []
    for epoch in range(params['epoch']):
        epoch_loss = loss_epoch(model, dataloader, criterion, optimizer, params, epoch)
        history.append(epoch_loss)
        
    return history

def Test(model, dataloader, params):  #############################
    tokenizer = params['tokenizer']
    max_len = params['max_len']
    vocab_size = tokenizer.vocab_size

    # model.eval()
    translated_sentences = []
    with torch.no_grad():
        for src_batch in tqdm(dataloader):
            src_batch = tokenizer(src_batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids
           
            output = model(params, src_batch, trg_vocab_size=vocab_size, teacher_forcing_ratio=0, inference=True)
            
            # Get the predicted token IDs
            _, predicted_ids = torch.max(output, dim=-1)
            
            # Convert to list and check for repetitive patterns
            for sentence_ids in predicted_ids:
                sentence_list = sentence_ids.tolist()
                
                # # 찾은 EOS 토큰 이후의 모든 토큰을 무시합니다.
                # if tokenizer.eos_token_id in sentence_list:
                #     sentence_list = sentence_list[:sentence_list.index(tokenizer.eos_token_id)]
                
                # 다시 텍스트로 변환
                translated_text = tokenizer.decode(sentence_list, skip_special_tokens=True)
                translated_sentences.append(translated_text)

    return translated_sentences



def main():
    # ==================================================
    # Step 1: 데이터 준비
    # Pandas DataFrame 형식으로 학습 데이터를 준비합니다.
    # 학습 데이터는 ecmData 변수에 저장되어야 합니다. 
    # 입력 데이터 예시
    ecmData = pd.read_csv('kor2.txt', sep='\t', encoding='cp949', header=None) 
    # ==================================================
    # Step 2: 데이터 전처리
    custom_DS = CustomDataset(ecmData)
    train_DL = torch.utils.data.DataLoader(custom_DS, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TranslationDataset(ecmData.iloc[:,0])  #############################
    test_DL = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    # ==================================================
    # Step 3: 모델 및 학습 설정
    BATCH_SIZE = 64
    EPOCH = 2
    max_len = 512
    drop_p = 0.1

    d_model = 128
    n_layers = 3

    d_hidden = 512 

    tokenizer = MarianTokenizer.from_pretrained(r"C:\Program Files\ECMiner\ECMiner_x64_withVOD_v.5.2.0.7592_20240719\Miniconda3\ModelSrc\Transformer_Transformer\model_directory")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id

    params = {
        'batch_size': BATCH_SIZE,
        'epoch': EPOCH,
        'max_len': max_len,
        'tokenizer': tokenizer,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'early_stopping' : EarlyStopping(verbose=True, target_loss = 0.0001)
    }

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    encoder = Encoder(vocab_size, d_model, d_hidden, n_layers, drop_p)
    decoder = Decoder(vocab_size, d_model, d_hidden, n_layers, drop_p)
    model = Seq2Seq(encoder, decoder, params['device']).to(params['device'])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # ==================================================
    # Step 4: 모델 학습
    train_losses = Train(model, train_DL, criterion, optimizer, params)

    # ==================================================
    # Step 5: 학습 결과 생성
    model.eval()
    all_translations = Test(model, test_DL, params)  #############################

    ecmData['YHAT'] = all_translations
    ecmData = ecmData['YHAT'].replace('', '!!!Translation failed due to insufficient model training!!!')
    # ==================================================
    # Step 6: 결과 및 모델 저장
    torch.save(model, "모델 저장 경로 입력")
    ecmData.to_csv("결과 저장 경로 입력", index = False, encoding = 'CP949')


if __name__ == "__main__":
    main()