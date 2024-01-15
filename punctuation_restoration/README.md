# BertPunc
The model is fine-tuned from a pretrained reimplementation of [BERT in Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT).

A punctation restoration model adds punctuation (e.g. period, comma, question mark) to an unsegmented, unpunctuated text. 


## Code

* train.py: training code
* data.py: helper function to read and transform data
* model.py: neural network model
* evaluate.py: evaluation 
