"""
Reference

url : https://deepblue-ts.co.jp/%E5%89%8D%E5%87%A6%E7%90%86/pytorch-dataset-transforms/
"""

import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import * 
from triming_textdataset import *

import copy

class MyTransform(object):
    def __init__(self, sentence, bos=False, eos=False, is_label=False):
        self.w2i = {}   # Word to Index
        self.i2w = {}   # Index to Word
        self.bos = bos  # did append symbol of beginning of sentence
        self.eos = eos  # did append symbol of end of sentence
        self.is_label = is_label

        # symbol char
        self.special_chars = [PAD_SYMBOL, BEGIN_SYMBOL, END_SYMBOL, UNKNOWN_SYMBOL, CONNECT_SYMBOL]
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]
        self.connect_char = self.special_chars[4]

        self.fit(sentence)     # create w2i and i2w from sentneces


    def __call__(self, sentence):
        return self.transform(sentence)  # return list of word ids
    
    def fit(self, sentences):
        self._words = set()

        # create "Unique" word list 
        for sentence in sentences:
            self._words.update(sentence[0].split(" "))
            if not self.is_label:
                self._words.update(sentence[1].split(" "))
        
        num_sp_char = len(self.special_chars)
        self.w2i = {}
        sp_count = 0
        for i,w in enumerate(self._words):
            if w in self.special_chars:
                sp_count += 1
                continue
            self.w2i[w] = (i+num_sp_char-sp_count)

        # create w2i from sentence
        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i

        # create i2w from w2i
        self.i2w = {i: w for w, i in self.w2i.items()}

    # create return value
    def transform(self, sentence):
        if self.bos:
            sentence = self.bos_char + " " + sentence
        if self.eos:
            sentence = sentence + " " + self.eos_char
        
        output = self.encode(sentence)
        return output
    

    # create list of word ids
    def encode(self, sentence):
        output = []
        for w in sentence.split(" "):
            if w in self.w2i:
                idx = self.w2i[w]
            else:
                idx = self.w2i[self.oov_char]
            output.append(idx)

        return output

    
    def encodes(self, sentences):
        output = []
        for sentence in sentences:
            temp = []
            for w in sentence.split(" "):
                if w in self.w2i:
                    idx = self.w2i[w]
                else:
                    idx = self.w2i[self.oov_char]
                temp.append(idx)
            output.append(temp)
            

        return torch.tensor(output)
    

    # create sentence from list of word ids
    def decode(self, sentnece):
        return [self.i2w[id] for id in sentnece]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length_s, max_length_t, transform=None, is_label=False):
        self.transform = transform
        self.max_length_s = max_length_s
        self.max_length_t = max_length_t
        self.data_num = len(data)
        self.data = data
        self.is_label = is_label
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = copy.deepcopy(self.data[idx])

        out_data[0] = self.transform.encode(out_data[0])
        out_data[0] = pad_sequences([out_data[0]], padding='post', maxlen=self.max_length_s)[0]
        out_data[0] = torch.LongTensor(out_data[0])
        if not self.is_label:
            out_data[1] = self.transform(out_data[1])
            out_data[1] = pad_sequences([out_data[1]], padding='post', maxlen=self.max_length_t)[0]
            out_data[1] = torch.LongTensor(out_data[1])
        else:
            out_data[1] = torch.LongTensor([out_data[1]])

        return out_data


def LoadSentenceData(data_path, transform=None, _shuffle=True):
    f = open(data_path, "r", encoding="utf-8")

    sentence = []
    max_length_s = 0
    max_length_t = 0

    temp_sentence = []

    # ファイルを読み込んでparam毎のリストを作成
    for line in f.readlines():
        line = line.rstrip("\n")

        if line == "<param>":
            temp_sentence.append([])
            continue

        temp_sentence[-1].append(line.split(" "))

    f.close()

    for param in temp_sentence:
        # 前後ずつのペアを作成
        for i in range(len(param) - 1):
            # 単語数の計算
            if IS_REVERSE:
                len_source = len(param[i+1])
                len_target = len(param[i])
            else:
                len_source = len(param[i])
                len_target = len(param[i+1])

            # 単語数で制限
            if (len_source < 10 or len_source > 25):
                continue
            if (len_target < 10 or len_target > 25):
                continue

            # 実際にリストに挿入
            if IS_REVERSE:
                sentence.append([ " ".join(param[i+1]), " ".join(param[i]) ])
            else:
                sentence.append([ " ".join(param[i]), " ".join(param[i+1]) ])

            # 最大長の更新
            if max_length_s < len_source:
                max_length_s = len_source
            if max_length_t < len_target:
                max_length_t = len_target

    print("num of sentences : {}".format(len(sentence)))

    if transform == None:
        transform = MyTransform(sentence, bos=False, eos=False, is_label=False)
    dataset = MyDataset(sentence, max_length_s, max_length_t, transform=transform, is_label=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=_shuffle, drop_last=True)
        
    return max_length_s, max_length_t, transform, dataset, dataloader



    
# create dataset values.
def LoadData(data_path, transform=None, _shuffle=True):
    f = open(data_path, "r", encoding="utf-8")
    sentence = []
    max_length_s = 0
    max_length_t = 0

    if IS_REVERSE:
        ind_s = 1
        ind_t = 0
    else:
        ind_s = 0
        ind_t = 1

    for line in f.readlines():
        line = line.rstrip("\n").split(",")
        s_length = len(line[ind_s])
        t_length = len(line[ind_t])

        if s_length > max_length_s:
            max_length_s = s_length
        if t_length > max_length_t:
            max_length_t = t_length
    
        sentence.append([" ".join(line[ind_s]), " ".join(line[ind_t])])
    f.close()

    # max_length_s += 2
    max_length_t += 2

    if transform == None:
        transform = MyTransform(sentence, bos=True, eos=True, is_label=False)
    dataset = MyDataset(sentence, max_length_s, max_length_t, transform=transform, is_label=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=_shuffle, drop_last=True)
        
    return max_length_s, max_length_t, transform, dataset, dataloader


def LoadLossData(data_path):
    f = open(data_path, "r", encoding="utf-8")
    sentence = []
    max_length_s = 0
    max_length_t = 0

    for line in f.readlines():
        line = line.rstrip("\n").split(",")

        line[0] = line[0].replace(BEGIN_SYMBOL, "b").replace(END_SYMBOL, "e").replace(PAD_SYMBOL, "p")
        line[1] = line[1].replace(BEGIN_SYMBOL, "b").replace(END_SYMBOL, "e").replace(PAD_SYMBOL, "p")

        s_length = len(line[0])+len(line[1])+1
        t_length = len(line[2])

        if s_length > max_length_s:
            max_length_s = s_length
        if t_length > max_length_t:
            max_length_t = t_length
    
        list0 = " ".join(line[0])
        list1 = " ".join(line[1])
        source = list0 + " ".join(["", CONNECT_SYMBOL, ""]) + list1
        source = source.replace("b", BEGIN_SYMBOL).replace("e", END_SYMBOL).replace("p", PAD_SYMBOL)
        sentence.append([source, int(line[2])])
    f.close()

    transform = MyTransform(sentence, bos=False, eos=False, is_label=True)
    dataset = MyDataset(sentence, max_length_s, max_length_t, transform=transform, is_label=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
    return max_length_s, max_length_t, transform, dataset, dataloader


def LoadEmbVec(file_path, transform, vocab_size):
    f = open(file_path, "r", encoding="utf-8")
    f.readline()

    emb_dict = {}

    emb_dict[transform.w2i[PAD_SYMBOL]] = torch.zeros(EMBEDDING_DIM)
    emb_dict[transform.w2i[CONNECT_SYMBOL]] = torch.ones(EMBEDDING_DIM)
    emb_dict[transform.w2i[BEGIN_SYMBOL]] = torch.ones(EMBEDDING_DIM)
    emb_dict[transform.w2i[BEGIN_SYMBOL]][0:64] = 0
    emb_dict[transform.w2i[END_SYMBOL]] = torch.ones(EMBEDDING_DIM)
    emb_dict[transform.w2i[END_SYMBOL]][65:128] = 0
    emb_dict[transform.w2i[UNKNOWN_SYMBOL]] = torch.ones(EMBEDDING_DIM)
    emb_dict[transform.w2i[UNKNOWN_SYMBOL]][129:192] = 0

    for line in f.readlines():
        datas = line.rstrip("\n").split(" ")
        word = datas[0]
        content = datas[1:]
        for i in range(len(content)):
            content[i] = float(content[i])
        emb_dict[transform.w2i[word]] = torch.tensor(content)
    f.close()

    out = torch.zeros(vocab_size, EMBEDDING_DIM)
    for i in range(vocab_size):
        out[i, :] = emb_dict[i]

    return out


def LoadTranslateData(mode="train", transform=None, _shuffle=True):
    dir_name = "data/translateEng2Jpn"
    f_eng = open("{}/{}.en".format(dir_name, mode), "r")
    f_jpn = open("{}/{}.ja".format(dir_name, mode), "r", encoding="UTF-8")

    sentence = []

    max_length_s = 0
    max_length_t = 0

    for eng in f_eng.readlines():
        eng = eng.rstrip("\n")

        jpn = f_jpn.readline()
        jpn = jpn.rstrip("\n")

        if IS_REVERSE:
            sentence.append([eng, jpn])
            len_source = len(eng.split(" "))
            len_target = len(jpn.split(" "))
        else:
            sentence.append([jpn, eng])
            len_source = len(jpn.split(" "))
            len_target = len(eng.split(" "))

        if len_source > max_length_s:
            max_length_s = len_source
        if len_target > max_length_t:
            max_length_t = len_target

    max_length_s += 2
    max_length_t += 2

    if transform == None:
        transform = MyTransform(sentence, bos=True, eos=True, is_label=False)
    dataset = MyDataset(sentence, max_length_s, max_length_t, transform=transform, is_label=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=_shuffle, drop_last=True)
        
    return max_length_s, max_length_t, transform, dataset, dataloader



if __name__ == "__main__":
    import torch.nn as nn

    max_length_s, max_length_t, transform, dataset, dataloader = LoadTranslateData()

    emb_dict = LoadEmbVec("data/word2vec/translate_row.vec.pt", transform, len(transform.w2i))

    embedding = nn.Embedding(len(transform.w2i), EMBEDDING_DIM, padding_idx=0)

    embedding.weight = nn.Parameter(emb_dict)

    print(embedding(torch.tensor([transform.w2i["went"]])) == emb_dict[transform.w2i["went"]])