import os
import torch
import datetime
from enum import Enum

# 学習率の変動仕様
class LRType(Enum):
    FIXED = "fixed"
    SEQUENCE = "sequence"
    STEP = "step"

"""
Constant value
"""
IS_COLAB = False    # flag of using GoogleColab

TIME_STAMP = str(datetime.date.today())
EXPERIENCE_NAME = "0EX_" + "emb_word2vec"
RUN_NAME = TIME_STAMP + "-" + EXPERIENCE_NAME

DEVICE = torch.device("cuda")

EMBEDDING_DIM = 192 # embedding word vector dimension num

HIDDEN_DIM = 512    # lstm hidden dimension num

BATCH_SIZE = 256   # mini batch size

MAIN_EPOCHS = 4000    # num epochs of main train
PRE_G_EPOCHS = 4000    # num epochs of generator pretrain
PRE_D_EPOCHS = 1500    # num epochs of discriminator pretrain

IS_PRETRAIN_G = True
IS_PRETRAIN_D = True

CLIP_RATE = 0.2     # rate of gradient clipping

DOES_USE_TEACHER_FORCING = False # flag of using teacher forcing
TEACHER_FORCING_RATE = 0      # rate of using teacher forcing

DISC_TRAIN_INTERVAL = 1      # discriminator train ratio

IS_REVERSE = False           # flag of using reverse data

IS_TRAIN_D_WHEN_TRAINING_G = False

BEGIN_SYMBOL = "<s>"        # symbol char, which mean begining of sentence.
END_SYMBOL = "</s>"         # symbol char, which mean ending of sentence.
PAD_SYMBOL = "<pad>"        # symbol char, which mean padding of sentence.
UNKNOWN_SYMBOL = "<unk>"    # symbol char, which mean unknown char.
CONNECT_SYMBOL = "<=?=>"    # connect symbol

SAMPLE_INTERVAL = 10   # output sample ratio of epoch

LR_MANAGEMENT_TYPE = LRType.FIXED   # flag of using "raise learning rate"

if LR_MANAGEMENT_TYPE == LRType.FIXED:
    START_LEARNING_RATE_G = 1e-4
    START_LEARNING_RATE_D = 1e-4
elif LR_MANAGEMENT_TYPE == LRType.SEQUENCE:
    LEARNING_RATE_OF_G_DICT = {          # dictionary of using learning rate
        1 :   1e-3,
        401 :   5*1e-4,
        501 :   1e-4,
        701 :   5*1e-5,
        901 :   1e-5
    }
    LEARNING_RATE_OF_D_DICT = {          # dictionary of using learning rate
        1 :   1e-6,
        51 :   1e-5,
        101 :   1e-4,
        151 :   1e-3,
        201 :   1e-2
    }

    START_LEARNING_RATE_G = LEARNING_RATE_OF_G_DICT[1]
    START_LEARNING_RATE_D = LEARNING_RATE_OF_D_DICT[1]
elif LR_MANAGEMENT_TYPE == LRType.STEP:
    START_LEARNING_RATE_G = 1e-3
    START_LEARNING_RATE_D = 1e-4
    DICRESE_RATE_OF_G_LR = 5e-8
    DICRESE_RATE_OF_D_LR = 1e-7

"""
Path data
"""
if IS_COLAB:
    # DATA_SET_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/duc_text.txt"                        # train dataset file path
    # TEST_SET_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/duc_text.txt"                         # test dataset file path
    DATA_SET_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/train_set_99.csv"                        # train dataset file path
    TEST_SET_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/train_set_99.csv"                         # test dataset file path
    DISC_TRAIN_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/train_loss_set.csv"
    DISC_TEST_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/test_loss_set.csv"
    EMB_VEC_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/data/emb_weight.vec"
else:
    DATA_SET_PATH = "data/duc_text.txt"                        # train dataset file path
    TEST_SET_PATH = "data/duc_text.txt"                         # test dataset file path
    # DATA_SET_PATH = "data/train_set_99.csv"                        # train dataset file path
    # TEST_SET_PATH = "data/train_set_99.csv"                         # test dataset file path
    DISC_TRAIN_PATH = "data/train_loss_set.csv"
    DISC_TEST_PATH = "data/test_loss_set.csv"
    EMB_VEC_PATH = "data/emb_weight.vec"


if IS_COLAB:
    now = datetime.datetime.now()
    CLOCK_STAMP = str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)
    LOG_LOSS_G_PATH = ["log", "loss_g.txt"]
    LOG_LOSS_D_PATH = ["log", "loss_d.txt"]
    LOG_ACC_TRAIN_PATH = ["log", "acc_train.txt"]
    LOG_ACC_TEST_PATH = ["log", "acc_test.txt"]
    LOG_INFOMATION_PATH = ["log", "infomation.txt"]
    LOG_NET_LOSS_0_PATH = ["log", "loss_net_0.txt"]
    LOG_NET_LOSS_1_PATH = ["log", "loss_net_1.txt"]
    LOG_NET_LOSS_2_PATH = ["log", "loss_net_2,txt"]
    LOG_NET_LOSS_TOTAL_PATH = ["log", "loss_net_total.txt"]
    LOG_GRADIENT_PENALTY_PATH = ["log", "gradient.txt"]
    LOG_BLEU_TRAIN_PATH = ["log", "bleu_train.txt"]
    LOG_BLEU_TEST_PAHT = ["log", "bleu_test.txt"]

    # separate forward translation or reverse translation
    if not IS_REVERSE:
        LOSS_GRAPH_PATH = ["results", "loss_graph.png"]
        ACC_GRAPH_PATH = ["results", "acc_graph.png"]
        SAMPLE_SENTENCE_PATH = ["results", "sample_sentence.txt"]
        LOSS_INFO_PATH = ["results", "loss_infomation.txt"]
        LOSS_NET_LABEL_GRAPH = ["results", "loss_net_label_graph.png"]
        LOSS_NET_GRAPH = ["results", "loss_net_graph.png"]
        GRADIENT_PENA_GRAPH = ["results", "gradient_graph.png"]
    else:
        LOSS_GRAPH_PATH = ["results", "loss_graph_reverse.png"]
        ACC_GRAPH_PATH = ["results", "acc_graph_reverse.png"]
        SAMPLE_SENTENCE_PATH = ["results", "sample_sentence_reverse.txt"]
        LOSS_INFO_PATH = ["results", "loss_infomation_reverse.txt"]
        LOSS_NET_LABEL_GRAPH = ["results", "loss_net_label_graph_reverse.png"]
        LOSS_NET_GRAPH = ["results", "loss_net_graph_reverse.png"]
        GRADIENT_PENA_GRAPH = ["results", "gradient_graph_reverse.png"]


    GENERATOR_MODEL_PATH = ["save_param", "gen_model.pth"]
    DISCRIMINATOR_MODEL_PATH = ["save_param", "disc_model.pth"]
    LOSS_NET_MODEL_PATH = ["save_param", "net_model.pth"]
    EMBEDDING_MODEL_PATH = ["save_param", "emb_model.pth"]
else:
    now = datetime.datetime.now()
    CLOCK_STAMP = str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)
    LOG_LOSS_G_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_g.txt"]
    LOG_LOSS_D_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_d.txt"]
    LOG_ACC_TRAIN_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "acc_train.txt"]
    LOG_ACC_TEST_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "acc_test.txt"]
    LOG_INFOMATION_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "infomation.txt"]
    LOG_NET_LOSS_0_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_0.txt"]
    LOG_NET_LOSS_1_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_1.txt"]
    LOG_NET_LOSS_2_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_2,txt"]
    LOG_NET_LOSS_TOTAL_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_total.txt"]
    LOG_GRADIENT_PENALTY_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "gradient.txt"]
    LOG_BLEU_TRAIN_PATH = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "bleu_train.txt"]
    LOG_BLEU_TEST_PAHT = ["log", TIME_STAMP+"-"+CLOCK_STAMP, "bleu_test.txt"]

    # separate forward translation or reverse translation
    if not IS_REVERSE:
        LOSS_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_graph.png"]
        ACC_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "acc_graph.png"]
        BLEU_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "bleu_graph.png"]
        SAMPLE_SENTENCE_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "sample_sentence.txt"]
        LOSS_INFO_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_infomation.txt"]
        LOSS_NET_LABEL_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_label_graph.png"]
        LOSS_NET_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_graph.png"]
        GRADIENT_PENA_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "gradient_graph.png"]
    else:
        LOSS_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_graph_reverse.png"]
        ACC_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "acc_graph_reverse.png"]
        BLEU_GRAPH_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "bleu_graph_reverse.png"]
        SAMPLE_SENTENCE_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "sample_sentence_reverse.txt"]
        LOSS_INFO_PATH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_infomation_reverse.txt"]
        LOSS_NET_LABEL_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_label_graph_reverse.png"]
        LOSS_NET_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "loss_net_graph_reverse.png"]
        GRADIENT_PENA_GRAPH = ["results", TIME_STAMP+"-"+CLOCK_STAMP, "gradient_graph_reverse.png"]


    GENERATOR_MODEL_PATH = ["save_param", TIME_STAMP+"-"+CLOCK_STAMP, "gen_model.pth"]
    DISCRIMINATOR_MODEL_PATH = ["save_param", TIME_STAMP+"-"+CLOCK_STAMP, "disc_model.pth"]
    LOSS_NET_MODEL_PATH = ["save_param", TIME_STAMP+"-"+CLOCK_STAMP, "net_model.pth"]
    EMBEDDING_MODEL_PATH = ["save_param", TIME_STAMP+"-"+CLOCK_STAMP, "emb_model.pth"]

if IS_COLAB:
    LOAD_GEN_MODEL_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/params/gen_model.pth"
    LOAD_DISC_MODEL_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/params/disc_model.pth"
    LOAD_NET_MODEL_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/params/net_model.pth"
    LOAD_EMB_MODEL_PATH = "/content/drive/My Drive/Seq2SeqGAN-Colab/params/emb_model.pth"
else:
    LOAD_GEN_MODEL_PATH = "params/gen_model.pth"
    LOAD_DISC_MODEL_PATH = "params/disc_model.pth"
    LOAD_NET_MODEL_PATH = "params/net_model.pth"
    LOAD_EMB_MODEL_PATH = "params/emb_model.pth"

IS_LOAD_GEN_MODEL = False   # flag of loading generator model
IS_LOAD_DISC_MODEL = False  # flag of loading discriminator model
IS_LOAD_NET_MODEL = False   # flag of loading loss_net model
IS_LOAD_EMB_MODEL = False   # flag of loading embedding model


# translate row path data(list) to input path data(str, joint)
def make_path(path_list, append=""):
    make_dir(path_list)
    if append == "":
        if IS_COLAB:
            return "/content/drive/My Drive/Seq2SeqGAN-Colab/results" + "/" + "/".join(path_list)
        else:
            return EXPERIENCE_NAME + "/" + "/".join(path_list)
    else:
        if IS_COLAB:
            out = "/content/drive/My Drive/Seq2SeqGAN-Colab/results" + "/" + "/".join(path_list)
        else:
            out = EXPERIENCE_NAME + "/" + "/".join(path_list)
        ind = out.find(".")
        temp = out[ind:-1]
        out = out.replace(temp, append+temp)
        return out


def make_dir(path_list):
    if IS_COLAB:
        dir_name = "/content/drive/My Drive/Seq2SeqGAN-Colab/results/" + EXPERIENCE_NAME
    else:
        dir_name = EXPERIENCE_NAME
    slash = "/"
    for i in range(len(path_list)-1):
        dir_name += slash + path_list[i]
    os.makedirs(dir_name, exist_ok=True)


def get_correct_label(device):
    # return 0.69 ~ 0.99
    return 0.69 + torch.rand(BATCH_SIZE, device=device)*3/10

def get_incorrect_label(device):
    # return 0.01 ~ 0.31
    return 0.01 + torch.rand(BATCH_SIZE, device=device)*3/10

def reshape_for_mmd(tensor):
    return tensor.view(BATCH_SIZE, 1, 1, -1)

def reshape_time_hms(ms):
    upper = int(ms)
    millisecond = int((ms - upper) * 1000)
    hour = int(upper // 3600)
    upper = upper % 3600
    minute = int(upper // 60)
    second = int(upper % 60)

    return "{}h {}m {}s {}ms".format(hour, minute, second, millisecond)


if __name__ == "__main__":
    print(reshape_time_hms(3000.435))