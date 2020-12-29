import tarfile
import re
import os
import sys
import string

from utils import *

def make_sentence_list(csv_file):
    count = 0
    max_length = 0
    out = []
    sentence = ""
    for data in csv_file.readlines():
        parse = data.split(",")
        count += 1
        if parse[3] == "。":
            sentence += parse[3]
            out.append(sentence)
            sentence = ""

            if count > max_length:
                max_length = count
            count = 0
        elif not (parse[3] == "「" or parse[3] == "」" or parse[3] == "　" or parse[3] == "（" or parse[3] == "？" or parse[3] == "！"):
            sentence += (parse[3] + " ")

    return out, max_length


def read_tar():
    print("make dataset file")

    root_dir_path = "C:/Users/YoppY/Documents/卒業研究/DUC_dataset/textfiles"
    if not os.path.exists(root_dir_path):
        print("root directory not exists")
        sys.exit()
    dir_path_list = os.listdir(root_dir_path)

    write_sentences = set()

    count = 0

    for dir_path in dir_path_list:
        if not os.path.exists(root_dir_path+"/"+dir_path):
            print("directory not exists")
            sys.exit()
        file_path_list = os.listdir(root_dir_path+"/"+dir_path)

        for file_path in file_path_list:
            print(file_path)
            file = open(root_dir_path+"/"+dir_path+"/"+file_path, "r")

            is_text = False
            lines = []

            try:
                sentence = "".join(file.readlines())
            except UnicodeDecodeError:
                continue
            finally:
                file.close()

            # p = re.compile(r"<text>[^>]*?</text>")
            sentence = sentence.lower()
            sentence = sentence.replace("\n", "")

            sentence = " ".join(re.findall(r"<text>[^>]*?</text>", sentence))
            sentence = sentence.replace("<text>", "")
            sentence = sentence.replace("</text>", "")
            sentence = sentence.replace("\'s", "")
            sentence = sentence.replace("\n", "")


            # for n in re.findall(r'\d*', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'\d* \d*', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'\d*-\d*', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'\d*.\d*', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'\d*st', sentence):
            #     sentence = sentence.replace(n, "NNth")
            # for n in re.findall(r'\d*nd', sentence):
            #     sentence = sentence.replace(n, "NNth")
            # for n in re.findall(r'\d*th', sentence):
            #     sentence = sentence.replace(n, "NNth")
            # for n in re.findall(r'\d*', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'NN \d+', sentence):
            #     sentence = sentence.replace(n, "NN")
            # for n in re.findall(r'\d+NN', sentence):
            #     sentence = sentence.replace(n, "NN")

            sentence = re.sub(r'[0-9]{8}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{7}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{6}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{5}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{4}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{3}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{2}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[0-9]{1}', ' NN ', sentence, flags=re.MULTILINE)
            sentence = sentence.replace("NN st", "NN th")
            sentence = sentence.replace("NN nd", "NN th")
            sentence = sentence.replace("<p>", "")
            sentence = sentence.replace("</p>", "")
            sentence = sentence.replace("dr.", "DR")
            sentence = sentence.replace("ms.", "MS")
            sentence = sentence.replace("j.s.", "JS")
            sentence = sentence.replace("j.s", "JS")
            sentence = sentence.replace("j. s.", "JS")
            sentence = sentence.replace("j. s", "JS")
            sentence = sentence.replace(" j.s.", "JS")
            sentence = sentence.replace(" j.s", "JS")
            sentence = sentence.replace("corp.", "CORP")
            sentence = sentence.replace("assn.", "ASSN")
            sentence = sentence.replace("a.m.", "AM")
            sentence = sentence.replace("p.m.", "PM")
            sentence = sentence.replace("mrs.", "MRS")
            sentence = sentence.replace("mr.", "MR")
            sentence = sentence.replace("inc.", "INC")
            sentence = sentence.replace("atty.", "ATTY")
            sentence = sentence.replace("u. s.", "US")
            sentence = sentence.replace("n.j.", "NJ")
            sentence = sentence.replace("u.n.", "UN")
            sentence = sentence.replace("n.j", "NJ")
            sentence = sentence.replace("u.n", "UN")
            sentence = sentence.replace("u.s.", "US")
            sentence = sentence.replace("u.s.a.", "USA")
            sentence = sentence.replace("u.s", "US")
            sentence = sentence.replace("u.s.a", "USA")
            sentence = sentence.replace("m.p.h.", "MPH")
            sentence = sentence.replace("m.p.h", "MPH")
            sentence = sentence.replace("b.c.", "BC")
            sentence = sentence.replace("b.c", "BC")
            sentence = sentence.replace("capt.", "CAPT")
            sentence = sentence.replace("a.m.e.", "AME")
            sentence = sentence.replace("a. m. e.", "AME")
            sentence = sentence.replace("v.m.d.", "VMD")
            sentence = sentence.replace("v. m. d.", "VMD")
            sentence = sentence.replace("ph.d.", "PHD")
            sentence = sentence.replace("ph. d.", "PHD")
            sentence = sentence.replace("k.l.", "KL")
            sentence = sentence.replace("k. l.", "KL")
            sentence = sentence.replace("d.c.", "DC")
            sentence = sentence.replace("D. C.", "DC")
            sentence = sentence.replace("ini .", "INI")
            sentence = sentence.replace("no.", "NO")
            sentence = sentence.replace("p.o.", "PO")
            sentence = sentence.replace("st.", "ST")
            sentence = sentence.replace("co.", "CO")
            sentence = sentence.replace("e.t.", "ET")
            sentence = sentence.replace("ave.", "AVE")
            sentence = sentence.replace("n.w.", "NW")
            sentence = sentence.replace("n.y.", "NY")
            sentence = sentence.replace("nev.", "NEV")
            sentence = sentence.replace("jr.", "JR")
            sentence = sentence.replace("calif.", "CALIF")
            sentence = sentence.replace("v. s.", "VS")
            sentence = sentence.replace("e.g.", "EG")
            sentence = sentence.replace("ft. worth", "FTWORTH")
            sentence = sentence.replace("mass.", "MASS")
            sentence = sentence.replace("n. j.", "NJ")
            sentence = sentence.replace("sr.", "SR")
            sentence = sentence.replace("conn.", "CONN")
            sentence = sentence.replace(". NN", " NN")

            for p in string.punctuation:
                if (p == ".") or (p == ",") or (p == "'"):
                    continue
                else:
                    sentence = sentence.replace(p, "")
            sentence = sentence.replace("'re", " are")
            sentence = sentence.replace("'s", " is")
            sentence = sentence.replace("'m", " am")
            sentence = sentence.replace("'ll", " will")
            sentence = sentence.replace("'ve", " have")
            sentence = sentence.replace("don't", "do not")
            sentence = sentence.replace("isn't", "is not")
            sentence = sentence.replace("aren't", "are not")
            sentence = sentence.replace("haven't", "is not")
            sentence = sentence.replace("won't", "will not")
            sentence = sentence.replace(" m r ", "")
            sentence = sentence.replace("nn", " nn ")
            sentence = sentence.replace("'", "")
            sentence = sentence.replace(",", " , ")
            sentence = sentence.replace(".......", ".")
            sentence = sentence.replace(". . . . . . .", ".")
            sentence = sentence.replace("......", ".")
            sentence = sentence.replace(". . . . . .", ".")
            sentence = sentence.replace(".....", ".")
            sentence = sentence.replace(". . . . .", ".")
            sentence = sentence.replace("....", ".")
            sentence = sentence.replace(". . . .", ".")
            sentence = sentence.replace("...", ".")
            sentence = sentence.replace(". . .", ".")
            sentence = sentence.replace("..", ".")
            sentence = sentence.replace(". .", ".")
            sentence = sentence.replace(".", " . ")

            sentence = " ".join([BEGIN_SYMBOL, sentence])
            sentence = sentence.replace(" . ", " ".join([" .", END_SYMBOL, BEGIN_SYMBOL, ""]))
            sentence = sentence.rstrip(" ").rstrip(BEGIN_SYMBOL)
            
            # sentence = sentence.replace("'", " '")
            sentence = sentence.replace("           ", " ")
            sentence = sentence.replace("       ", " ")
            sentence = sentence.replace("   ", " ")
            sentence = sentence.replace("               ", " ")
            sentence = sentence.replace("              ", " ")
            sentence = sentence.replace("             ", " ")
            sentence = sentence.replace("            ", " ")
            sentence = sentence.replace("           ", " ")
            sentence = sentence.replace("          ", " ")
            sentence = sentence.replace("         ", " ")
            sentence = sentence.replace("        ", " ")
            sentence = sentence.replace("       ", " ")
            sentence = sentence.replace("      ", " ")
            sentence = sentence.replace("     ", " ")
            sentence = sentence.replace("    ", " ")
            sentence = sentence.replace("   ", " ")
            sentence = sentence.replace("  ", " ")

            sentence = sentence.lower()
            if (re.match(r"^['., ' ]+$", sentence) is None):
                if (sentence != ""):
                    sentence = sentence.strip()
                    write_sentences.add(sentence+"\n")
                    count += 1

    write_file = open("data/ducdata.txt", "w")
    write_file.writelines(write_sentences)
    write_file.close()
    print("num of sentences :", count)
    print("Done.")

if __name__ == "__main__":
    read_tar()