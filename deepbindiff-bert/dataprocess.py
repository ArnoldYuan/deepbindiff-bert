import pickle
from transformers import BertTokenizer
import os

def del_blank(token):
    return token.replace(' ', '')

def gen_dic(filename, outputname):
    with open(filename, 'rb') as file:
        dic = pickle.load(file)
    with open(outputname, 'w') as file:
        spec_tokens = ['[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for token in spec_tokens:
            file.write(token + '\n')
        for value in dic.values():
            file.write(del_blank(value) + '\n')

        # load vocabulary from bert-base-uncased
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for item in list(tokenizer.vocab.keys())[739:]:
            file.write(item + '\n')

def gen_corpus(fn1, fn2, outputname):
    with open(fn1, 'rb') as f1:
        walks = pickle.load(f1)
    with open(fn2, 'rb') as f2:
        blockIdxToTokens = pickle.load(f2)
    with open(outputname, 'w') as file:
        for walk in walks:
            sentence = ""
            for idx in walk:
                tokens = blockIdxToTokens[idx]
                for token in tokens:
                    sentence += del_blank(token)
                    sentence += ' '
            file.write(sentence.strip() + '\n')

def gen_dic_all(dirname, outputname):
    subdirs = os.listdir(dirname)
    output = open(outputname, 'w')
    spec_tokens = ['[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for token in spec_tokens:
        output.write(token + '\n')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for item in list(tokenizer.vocab.keys())[999:]:
        output.write(item + '\n')
    diclist = []
    for subdir in subdirs:
        if not os.path.isdir(dirname + '/' + subdir):
            continue
        files = os.listdir(dirname + '/' + subdir)
        if not ('walks.pkl' in files and 'dictionary.pkl' in files and 'bb2token.pkl' in files):
            continue
        with open(dirname + '/' + subdir + '/' + 'dictionary.pkl', 'rb') as file:
            dic = pickle.load(file)
        for value in dic.values():
            if value not in diclist:
                diclist.append(value)
    for value in diclist:
        output.write(del_blank(value) + '\n')
    output.close()

def gen_corpus_all(dirname, outputname):
    subdirs = os.listdir(dirname)
    output = open(outputname, 'w')
    for subdir in subdirs:
        if not os.path.isdir(dirname + '/' + subdir):
            continue
        files = os.listdir(dirname + '/' + subdir)
        if not ('walks.pkl' in files and 'dictionary.pkl' in files and 'bb2token.pkl' in files):
            continue
        with open(dirname + '/' + subdir + '/' + 'walks.pkl', 'rb') as f1:
            walks = pickle.load(f1)
        with open(dirname + '/' + subdir + '/' + 'bb2token.pkl', 'rb') as f2:
            blockIdxToTokens = pickle.load(f2)
        with open(outputname, 'w') as file:
            for walk in walks:
                sentence = ""
                for idx in walk:
                    tokens = blockIdxToTokens[idx]
                    for token in tokens:
                        sentence += del_blank(token)
                        sentence += ' '
                output.write(sentence.strip() + '\n')
        output.write('\n')
    output.close()

# gen_dic('data/1/dictionary.pkl', 'vocab/vocab.txt')
# gen_corpus('data/1/walks.pkl', 'data/1/bb2token.pkl', 'data/corpus.txt')
if __name__ == '__main__':
    gen_dic_all('data', 'vocab/vocab.txt')
    gen_corpus_all('data', 'data/corpus.txt')

