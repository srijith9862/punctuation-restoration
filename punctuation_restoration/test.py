import pandas as pd
import numpy as np
from glob import glob
from pytorch_pretrained_bert import BertTokenizer
import torch
from torch import nn
%matplotlib inline
import json
from tqdm import tqdm
from sklearn import metrics

from model import BertPunc
from data import load_file, preprocess_data, create_data_loader
glob('models/*')
path = 'models/20240115_225353/'
data_test = load_file('data/text.txt')
# data_test = load_file('data/NPR-podcasts/test')
with open(path+'hyperparameters.json', 'r') as f:
    hyperparameters = json.load(f)
hyperparameters
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

punctuation_enc = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3
}

segment_size = hyperparameters['segment_size']
X_test, y_test = preprocess_data(data_test, tokenizer, punctuation_enc, segment_size)
output_size = len(punctuation_enc)
dropout = hyperparameters['dropout']
bert_punc = nn.DataParallel(BertPunc(segment_size, output_size, dropout).cuda())
progress = pd.read_csv(path+'progress.csv', delimiter=';')
bert_punc.load_state_dict(torch.load(path+'model'))
bert_punc.eval();
batch_size = 1024
data_loader_test = create_data_loader(X_test, y_test, False, batch_size)
def predictions(data_loader):
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            output = bert_punc(inputs)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
    return y_pred, y_true
def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[1, 2, 3])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[1, 2, 3])
    result = pd.DataFrame(
        np.array([precision, recall, f1]), 
        columns=list(punctuation_enc.keys())[1:], 
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result


y_pred_test, y_true_test = predictions(data_loader_test)
eval_test = evaluation(y_pred_test, y_true_test)
print(eval_test)
