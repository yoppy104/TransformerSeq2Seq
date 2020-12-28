import matplotlib.pyplot as plt
from utils import *
import time

import datetime

def output_graph(losses, file_name, ylim=0):
    plt.figure(figsize=(12,6))
    plt.plot(range(len(losses)), losses)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.title('Encoder Loss')
    if ylim != 0:
        plt.ylim(ylim)
    plt.savefig(file_name)
    plt.clf()

def output_dual_graph(graphs, legend, file_name, ylim=0):
    plt.figure(figsize=(6, 6))
    cols = ["b", "r", "g"]
    for i in range(len(graphs)):
        plt.plot(range(len(graphs[i])), graphs[i], c=cols[i])
    plt.xlabel('EPOCH')
    plt.ylabel('')
    if ylim != 0:
        plt.ylim(ylim)
    plt.title('Compare Graph')
    plt.legend(legend)
    plt.savefig(file_name)
    plt.clf()

def output_single_sentence(sentences, file_name):
    f = open(file_name, "w")
    for sentence in sentences:
        f.write(sentence + "\n")
    f.close()
    

def output_sentence(sentences, file_name):
    f = open(file_name, "w", encoding="UTF-8")
    for sentence in sentences:
        source = sentence[0]
        generated = sentence[1]
        target = sentence[2]
        f.write("source   : {0}\ntarget   : {1}\ngenerate : {2}\n\n".format(source, target, generated))
    f.close()

def calc_accuracy(sentences):
    acc = 0
    count = 0
    for sentence in sentences:
        if IS_REVERSE:
            fnc = sentence[1]
            ans = sentence[0]
        else:
            fnc = sentence[0]
            ans = sentence[1]

        fnc = remove_after_the_end_symbol(fnc).replace(BEGIN_SYMBOL,"").replace(END_SYMBOL,"").replace(PAD_SYMBOL,"").replace(UNKNOWN_SYMBOL,"")
        ans = remove_after_the_end_symbol(ans).replace(BEGIN_SYMBOL,"").replace(END_SYMBOL,"").replace(PAD_SYMBOL,"").replace(UNKNOWN_SYMBOL,"")
        
        count += 1
        correct = calc(fnc)
        if correct == ans:
            acc += 1
    if count == 0:
        return 0
    return acc / count * 100

def calc_label(sentences):
    out_labels = torch.zeros(BATCH_SIZE, dtype=torch.long)
    count_label = [0, 0, 0]
    count = 0
    for sentence in sentences:
        if IS_REVERSE:
            fnc = sentence[1]
            ans = sentence[0]
        else:
            fnc = sentence[0]
            ans = sentence[1]

        fnc = remove_after_the_end_symbol(fnc).replace(BEGIN_SYMBOL,"").replace(END_SYMBOL,"").replace(PAD_SYMBOL,"").replace(UNKNOWN_SYMBOL,"")
        ans = remove_after_the_end_symbol(ans).replace(BEGIN_SYMBOL,"").replace(END_SYMBOL,"").replace(PAD_SYMBOL,"").replace(UNKNOWN_SYMBOL,"")

        fnc = fnc.split("+")
        if (len(fnc) != 2) or (not fnc[0].isdigit()) or (not fnc[1].isdigit()):
            out_labels[count] = 2
            count += 1
            count_label[2] += 1
            continue
        
        calc_ans = int(fnc[0]) + int(fnc[1])
        if str(calc_ans) == str(ans):
            out_labels[count] = 0
            count_label[0] += 1
        else:
            out_labels[count] = 1
            count_label[1] += 1

        count += 1
    
    print(count_label)
    return out_labels
    

# calculate char formula
def calc(sentence):
    if "+" in sentence:
        sentence = sentence.split("+")
        if (len(sentence) != 2) or (not sentence[0].isdigit()) or (not sentence[1].isdigit()):
            return "nan"
        ans = int(sentence[0]) + int(sentence[1])
        return str(ans)
    elif "-" in sentence:
        sentence = sentence.split("-")
        if (len(sentence) != 2) or (not sentence[0].isdigit()) or (not sentence[1].isdigit()):
            return "nan"
        ans = int(sentence[0]) - int(sentence[1])
        return str(ans)
    else:
        return "nan"


# remove after the eos from input a sentence
def remove_after_the_end_symbol(sentence):
    return sentence
    ind = sentence.find(END_SYMBOL)
    if (ind != -1):
        sentence = sentence[0:ind]
    return sentence


def output_log_data(path_list, datas):
    fname = make_path(path_list)
    f = open(fname, "w")
    for data in datas:
        f.write(str(data)+"\n")
    f.close()


def generate_graph(file_paths, out_name, legend=[], ylim=0):
    datas = []
    
    for file_path in file_paths:
        datas.append([])
        f = open(file_path, "r")
        count = 0
        for line in f.readlines():
            count += 1
            datas[-1].append(float(line.strip("\n")))
        f.close()
    
    if len(datas) == 1:
        output_graph(datas[0], out_name, ylim=ylim)
    else:
        output_dual_graph(datas, legend, out_name, ylim=ylim)


class Infomation:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

        self.interval = 0

        self.pretrain_g_run_time = 0
        self.pretrain_d_run_time = 0
        self.main_train_run_time = 0
    
    def start_clock(self):
        self.start_time = time.time()

    def lap_time(self):
        temp_time = time.time() - self.start_time
        if self.interval == 0:
            self.pretrain_g_run_time = temp_time
        elif self.interval == 1:
            self.pretrain_d_run_time = temp_time - self.pretrain_g_run_time
        else:
            self.main_train_run_time = temp_time - self.pretrain_g_run_time - self.pretrain_d_run_time
        self.interval += 1
    
    def finish_clock(self):
        self.end_time = time.time()

    def output_infomation(self):
        file_path = make_path(LOG_INFOMATION_PATH)
        out_file = open(file_path, "w")

        # output user's memo
        out_file.write("<< infomation >>\n\n\n")

        # output experience's infomation
        out_file.write("<< experience infomation >>\n")
        out_file.write("experience name     : {}\n".format(EXPERIENCE_NAME))
        out_file.write("experience mode     : {}\n".format("Backward" if IS_REVERSE else "Forward"))
        out_file.write("experience date     : {}\n".format(TIME_STAMP))
        out_file.write("pretrain g run time : {}\n".format(reshape_time_hms(self.pretrain_g_run_time)))
        out_file.write("pretrain d run time : {}\n".format(reshape_time_hms(self.pretrain_d_run_time)))
        out_file.write("main train run time : {}\n".format(reshape_time_hms(self.main_train_run_time)))
        out_file.write("total run time      : {}\n".format(reshape_time_hms(self.end_time-self.start_time)))
        out_file.write("\n")

        # output parameter's infomation
        out_file.write("<< parameters >>\n")
        out_file.write("main epochs                     : {}\n".format(MAIN_EPOCHS))
        out_file.write("pretrain g epochs               : {}\n".format(PRE_G_EPOCHS if IS_PRETRAIN_G else "None"))
        out_file.write("pretrain d epochs               : {}\n".format(PRE_D_EPOCHS if IS_PRETRAIN_D else "None"))
        out_file.write("\n")
        out_file.write("embedding dimension             : {}\n".format(EMBEDDING_DIM))
        out_file.write("hidden dimension                : {}\n".format(HIDDEN_DIM))
        out_file.write("batch size                      : {}\n".format(BATCH_SIZE))
        if DOES_USE_TEACHER_FORCING:
            out_file.write("use teacherforcing              : {}[rate-{}]\n".format(DOES_USE_TEACHER_FORCING, TEACHER_FORCING_RATE))
        else:
            out_file.write("use teacherforcing              : {}\n".format(DOES_USE_TEACHER_FORCING))
        out_file.write("rate of discriminator training  : {}\n".format(DISC_TRAIN_INTERVAL))
        out_file.write("is train d when training g      : {}\n".format(IS_TRAIN_D_WHEN_TRAINING_G))
        out_file.write("\n")

        # output learning rate transition
        out_file.write("<< transition of learning rate >>\n")
        out_file.write("lr management type : {}\n".format(LR_MANAGEMENT_TYPE))
        out_file.write("generator lr       : ")
        if LR_MANAGEMENT_TYPE==LRType.SEQUENCE:
            space = ""
            for key in LEARNING_RATE_OF_G_DICT.keys():
                out_file.write("{}[{: >4} epoch]  {}\n".format(space, key, LEARNING_RATE_OF_G_DICT[key]))
                space = "                     "
        else:
            out_file.write("{}\n".format(START_LEARNING_RATE_G))
            if LR_MANAGEMENT_TYPE == LRType.STEP:
                out_file.write("dicrese rate : {}\n\n".format(DICRESE_RATE_OF_G_LR))
        
        out_file.write("discriminator lr   : ")
        if LR_MANAGEMENT_TYPE==LRType.SEQUENCE:
            space = ""
            for key in LEARNING_RATE_OF_D_DICT.keys():
                out_file.write("{}[{: >4} epoch]  {}\n".format(space, key, LEARNING_RATE_OF_D_DICT[key]))
                space = "                     "
        else:
            out_file.write("{}\n".format(START_LEARNING_RATE_D))
            if LR_MANAGEMENT_TYPE == LRType.STEP:
                out_file.write("dicrese rate       : {}\n\n".format(DICRESE_RATE_OF_D_LR))
        

def output_results():
    generate_graph(
        [
            make_path(LOG_LOSS_G_PATH)
        ], 
        make_path(LOSS_GRAPH_PATH), 
        legend=["generator", "discriminator"]
    )
    generate_graph(
        [
            make_path(LOG_ACC_TRAIN_PATH), 
            make_path(LOG_ACC_TEST_PATH)
        ], 
        make_path(ACC_GRAPH_PATH, append="_limited"), 
        legend=["train", "test"], 
        ylim=(0, 100)
    )
    generate_graph(
        [
            make_path(LOG_ACC_TRAIN_PATH), 
            make_path(LOG_ACC_TEST_PATH)
        ], 
        make_path(ACC_GRAPH_PATH), 
        legend=["train", "test"]
    )
        



if __name__ == "__main__":
    dir_name = ""
    if dir_name != "":
        import utils
        utils.EXPERIENCE_NAME = dir_name
    date_name = "2020-12-27-23_5_44"
    LOG_LOSS_G_PATH[1] = date_name
    LOG_LOSS_D_PATH[1] = date_name
    LOG_ACC_TRAIN_PATH[1] = date_name
    LOG_ACC_TEST_PATH[1] = date_name

    LOG_NET_LOSS_TOTAL_PATH[1] = date_name
    LOG_NET_LOSS_0_PATH[1] = date_name
    LOG_NET_LOSS_1_PATH[1] = date_name
    LOG_NET_LOSS_2_PATH[1] = date_name
    LOSS_NET_LABEL_GRAPH[1] = date_name
    LOSS_NET_GRAPH[1] = date_name
    LOG_BLEU_TEST_PAHT[1] = date_name
    LOG_BLEU_TRAIN_PATH[1] = date_name

    LOSS_GRAPH_PATH[1] = date_name
    ACC_GRAPH_PATH[1] = date_name
    BLEU_GRAPH_PATH[1] = date_name

    generate_graph(
        [
            make_path(LOG_LOSS_G_PATH)
            # make_path(LOG_LOSS_D_PATH)
        ], 
        make_path(LOSS_GRAPH_PATH), 
        legend=["generator", "discriminator"]
    )
    generate_graph(
        [
            make_path(LOG_BLEU_TRAIN_PATH), 
            make_path(LOG_BLEU_TEST_PAHT)
        ], 
        make_path(BLEU_GRAPH_PATH), 
        legend=["train", "test"]
    )
    generate_graph(
        [
            make_path(LOG_ACC_TRAIN_PATH), 
            make_path(LOG_ACC_TEST_PATH)
        ], 
        make_path(ACC_GRAPH_PATH, append="_limited"), 
        legend=["train", "test"], 
        ylim=(0, 100)
    )
    generate_graph(
        [
            make_path(LOG_ACC_TRAIN_PATH), 
            make_path(LOG_ACC_TEST_PATH)
        ], 
        make_path(ACC_GRAPH_PATH), 
        legend=["train", "test"]
    )
