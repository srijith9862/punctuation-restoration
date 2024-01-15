import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import string 
import csv

def get_label(punctuation):
    if punctuation == ',':
        return '1'
    elif punctuation == '.':
        return '2'
    elif punctuation == '?':
        return '3'
    else:
        return '0'

# def preprocess(fpath):
file_path = 'train.csv'
# Read CSV file into a DataFrame
df = pd.read_csv(file_path)
df['Response'] = df['Response'].replace('\n', ' ', regex=True)
unique_paragraphs = df['Response'].unique().tolist()
unique_paragraphs = [paragraph for paragraph in unique_paragraphs if pd.notna(paragraph)]
print("No.of unique paragraphs:", len(unique_paragraphs))
# Splitting the sentences in 80:20 ratio
train_paragraphs, test_paragraphs = train_test_split(unique_paragraphs, test_size=0.2, random_state=10)
train_paragraphs, dev_paragraphs = train_test_split(train_paragraphs, test_size=0.2, random_state=15)
# tokenized_sentences = [sent_tokenize(paragraph) for paragraph in unique_paragraphs]
# sentences = [sentence for paragraph_sentences in tokenized_sentences for sentence in paragraph_sentences]
# print("No.of sentences:", len(sentences))
# sentences = list(set(sentences))
# print("No.of unique sentences:", len(sentences))
# Remove punctuation from the train sentences and keep only allowed punctuation
allowed_punctuation = {'.', ',', '?'}
result = []
for paragraph in train_paragraphs:
    for word in paragraph.split(' '):
        try:
            word = word.rstrip('\n')
            if word[-1] in allowed_punctuation: 
                label = get_label(word[-1])
                word = word[:-1]
            else:
                label = '0'
            result.append((word,' ', label))
        except:
            continue
with open('train.txt', 'w', newline='') as txtfile:
    for entry in result:
        txtfile.write(f'{entry[0]}\t{entry[2]}\n')
result = []
for paragraph in test_paragraphs:
    for word in paragraph.split(' '):
        try:
            word = word.rstrip('\n')
            if word[-1] in allowed_punctuation: 
                label = get_label(word[-1])
                word = word[:-1]
            else:
                label = '0'
            result.append((word,' ', label))
        except:
            continue
with open('test.txt', 'w', newline='') as txtfile:
    for entry in result:
        txtfile.write(f'{entry[0]}\t{entry[2]}\n')
result = []
for paragraph in dev_paragraphs:
    for word in paragraph.split(' '):
        try:
            word = word.rstrip('\n')
            if word[-1] in allowed_punctuation: 
                label = get_label(word[-1])
                word = word[:-1]
            else:
                label = '0'
            result.append((word,' ', label))
        except:
            continue
with open('dev.txt', 'w', newline='') as txtfile:
    for entry in result:
        txtfile.write(f'{entry[0]}\t{entry[2]}\n')
# for paragraph in test_paragraphs[:1]:
#     for word in paragraph.split(' '):
#         try:
#             if word[-1] in allowed_punctuation: 
#                 label = get_label(word[-1])
#                 word = word[:-1]
#             else:
#                 label = '0'
#             result.append((word,' ', label))
#         except:
#             continue
# with open('output.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['word', 'Label'])
#     for i in result:
#         # rows = process_sentence(sentence.strip())
#         csv_writer.writerows(result)
exit(0)
    # for sentence in sentences:
        # sentence_no_punct = ''.join(char if char in allowed_punctuation or char.isalnum() else ' ' for char in sentence)
        # sentences_allowed_punct.append(sentence_no_punct)
    # sentences = sentences_allowed_punct
    # punctuation_count = 0
    # removed_punctuation_set = set()
    # Remove punctuation from the sentences and count instances
    # sentences_no_punct = []
    # for sentence in sentences:
        # sentence_no_punct = ''.join(char if char not in string.punctuation else ' ' for char in sentence)
        # punctuation_count += sum(1 for char in sentence if char in string.punctuation)
        # removed_punctuation_set.update(char for char in sentence if char in string.punctuation)
        # sentences_no_punct.append(sentence_no_punct)
# 
    # print("Removed punctuation marks:", removed_punctuation_set)
    # print("No.of punctations removed:",punctuation_count)
# 
# 
    # Writing train sentences to a file
    # with open('sentences.txt', 'w', encoding='utf-8') as f:
        # for sentence in sentences:
            # f.write(sentence + '\n')
    # Writing train sentences to a file
    # with open('sentences_no_punct.txt', 'w', encoding='utf-8') as f:
        # for sentence in sentences_no_punct:
            # f.write(sentence + '\n')
    # return sentences, sentences_no_punct

# def preprocess(fpath):
#     file_path = fpath

#     # Read CSV file into a DataFrame
#     df = pd.read_csv(file_path)

#     # Remove newline characters from the data columns
#     # df['Context'] = df['Context'].replace('\n', ' ', regex=True)
#     df['Response'] = df['Response'].replace('\n', ' ', regex=True)

#     # Get unique paragraphs for 'context' column
#     # unique_context_paragraphs = df['Context'].unique().tolist()

#     # Get unique paragraphs for 'response' column
#     unique_paragraphs = df['Response'].unique().tolist()

#     # Get all the unique paragraphs in the data
#     # unique_paragraphs = unique_context_paragraphs + unique_response_paragraphs

#     # To remove the nan values from the data
#     unique_paragraphs = [paragraph for paragraph in unique_paragraphs if pd.notna(paragraph)]
#     print("No.of unique paragraphs:", len(unique_paragraphs))


#     # Tokenize paragraphs into sentences
#     tokenized_sentences = [sent_tokenize(paragraph) for paragraph in unique_paragraphs]

#     # Combine the list of sentences lists into a single list of sentences
#     sentences = [sentence for paragraph_sentences in tokenized_sentences for sentence in paragraph_sentences]
#     print("No.of sentences:", len(sentences))

#     # Getting the unique sentences for further training and testing
#     sentences = list(set(sentences))
#     print("No.of unique sentences:", len(sentences))
#     # return sentences
#     # Splitting the sentences in 80:20 ratio
#     train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=10)
#     print("No.of train sentences:",len(train_sentences))
#     print("No.of test sentences:",len(test_sentences))

#     # Remove punctuation from the test sentences
#     # test_sentences_no_punct = [''.join(char for char in sentence if char not in string.punctuation) for sentence in test_sentences]


#     # Remove punctuation from the train sentences and keep only allowed punctuation
#     train_sentences_no_punct = []
#     allowed_punctuation = {'.', ',', '?'}
#     for sentence in train_sentences:
#         sentence_no_punct = ''.join(char if char in allowed_punctuation or char.isalnum() else ' ' for char in sentence)
#         train_sentences_no_punct.append(sentence_no_punct)
    

#     train_sentences = train_sentences_no_punct

#     punctuation_count = 0
#     removed_punctuation_set = set()
#     # Remove punctuation from the test sentences and count instances
#     test_sentences_no_punct = []
#     for sentence in test_sentences:
#         sentence_no_punct = ''.join(char if char in allowed_punctuation or char.isalnum() else ' ' for char in sentence)
#         test_sentences_no_punct.append(sentence_no_punct)
#     test_sentences = test_sentences_no_punct
#     test_sentences_no_punct = []
#     for sentence in test_sentences:
#         sentence_no_punct = ''.join(char if char not in string.punctuation else '' for char in sentence)
#         punctuation_count += sum(1 for char in sentence if char in string.punctuation)
#         removed_punctuation_set.update(char for char in sentence if char in string.punctuation)
#         test_sentences_no_punct.append(sentence_no_punct)

#     print("Removed punctuation marks:", removed_punctuation_set)
#     print("No.of punctations removed:",punctuation_count)

    
#     train_sentences, validate_sentences = train_test_split(train_sentences, test_size=0.2, random_state=10)
#     # Writing train sentences to a file
#     with open('train_sentences.txt', 'w', encoding='utf-8') as f:
#         for sentence in train_sentences:
#             f.write(sentence + '\n')


#     with open('validate_sentences.txt', 'w', encoding='utf-8') as f:
#         for sentence in validate_sentences:
#             f.write(sentence + '\n')


#     # Writing test sentences to a file
#     with open('test_sentences.txt', 'w', encoding='utf-8') as f:
#         for sentence in test_sentences:
#             f.write(sentence + '\n')
    
#     # Writing test sentences without punctuation to a file
#     with open('test_sentences_no_punct.txt', 'w', encoding='utf-8') as f:
#         for sentence in test_sentences_no_punct:
#             f.write(sentence + '\n')
#     return train_sentences, test_sentences, test_sentences_no_punct


# if __name__ == '__main__':
#     # train_sentences, test_sentences, test_sentences_no_punct = preprocess('train.csv')
#     sentences, sentences_no_punct = preprocess('train.csv')