import pandas as pd
import re, unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class QueryrelevanceClassifier(nn.Module):
    def __init__(self, freeze_bert = True):#, num_labels = 4):
        super(QueryrelevanceClassifier, self).__init__()
        #self.num_labels = num_labels# Multi-label
        #initialising
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Fressing the weights of BERT layer, computationaly cheap to train only the classification layer  
        
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.classifier = nn.Linear(768, 1)#num_labels

    def forward(self, sent, attn_masks):
        cont_reps, _ = self.bert_layer(sent, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]#class rep#[cls] 
        
        #Feeding cls_rep to the classifier layer
        logits = self.classifier(cls_rep) #
        #probs = nn.softmax(cls_rep)#probs   
        return logits

def pre_processing(s):
    #remove numbers, puncuation, change encos=ding
    str_ = re.sub(r'\d+', '', s.strip().lower())# @ number to word format
    str_ = str_.translate(table)#can keep ':.
    str_ = unicodedata.normalize('NFKD', str_).encode('ascii', 'ignore').decode('ascii', 'ignore')
    print(str_)
    return  str_

def BERT_tokeniser(sentence_list):
    tokens_list = [tokenizer.tokenize(sen) for sen in sentence_list]
    tokens = []
    for i, token in enumerate(tokens_list):
        if i  == 0:
            tokens.append('[CLS]')
        tokens.extend(token)
        if i != len(tokens_list):
            tokens.append('[SEP]')
        
    padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]#padding
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = []
    value = 0
    count = 0
    for t in padded_tokens:
        if t == '[SEP]' and count == 0:
            value == 1
            count == 1
        elif t == '[SEP]' and count == 1:
            value == 0
        seg_ids.append(value)
    
    #BERT Vocabulary index
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    
    #Converting to torch tensors after tokenisation
    token_ids = torch.tensor(token_ids).unsqueeze(0)#[1 x T]#adding dimnesion to tensor
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0) 

    return  token_ids, attn_mask, seg_ids


def train_model(model, criterion, optimiser, train_set):
    for ep in range(50):      
        for i, data in enumerate(train_set):
            model.train()
            sentence, attn_mask, labels = train[i]
            optimiser.zero_grad()  
           
            #Loss computattion
            logits = model(sentence, attn_mask)
            loss = criterion(logits.squeeze(-1), labels.float())#squueze --> removing dim of tensor [1]
            #probs(will be computed for all pred 0,m with y and summed)
            #one hot for the label variable1

            #Backpropagation
            loss.backward()
            optimiser.step()

        if (i) % 10 == 0:
            print("Iteration {} of epoch {} complete. Loss : {}".format(i+1, ep+1, loss.item()))

#variables
table = str.maketrans(dict.fromkeys('!"#$%&\():.*+,-/;<=>?@[\\]^_`{|}~'))    
T= 200
def main():#main function
    #Importing and pre processing data
    df = pd.read_csv('data/crowdflower-search-relevance/train.csv')
    df_test = pd.read_csv('data/crowdflower-search-relevance/test.csv')
    
    train_set = []
    test_set = []
    for i in range(len(df)):
        sentences = [df['query'][i], df['product_title'][i], df['product_description'][i] if type(df['product_description'][i]) is str else '']
        process = [pre_processing(s) for s in sentences]
        token_id, attn_mask, seg_id = BertTokenizer(process)
        label = torch.tensor(df['median_relevance'][i])
        train_set.append([token_id,attn_mask,label])
        
        
    model = QueryrelevanceClassifier(freeze_bert = True)
    criterion = nn.CrossEntropyLoss()#for multi-class
    optimiser = optim.Adam(model.parameters(), lr = 2e-5)
    
    #Model training
    train_model(model, criterion, optimiser, train_set)
    
    #Model testing
    queryID = df_test.id.tolist(), preds = []
    for data in test_set:
        sentence, attn_mask = test_set[i]
        pred = model(sentence, attn_mask)
        preds.append(pred)
    
    df_pred = pd.DataFrame(zip(queryID, preds), columns=['id', 'prediction'])
        
    
        
    
loss = nn.CrossEntropyLoss()
input = torch.randn(1, 5, requires_grad=True)
target = torch.empty(1, dtype=torch.long).random_(2)
output = loss(input, target)
output.backward()
    
    
    
        