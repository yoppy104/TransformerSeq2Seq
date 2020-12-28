import random

from utils import *

def selectRandomNumber(is_hard=True):
    if is_hard: 
        val1 = random.randint(0, 999)
        val2 = random.randint(0, 999)
    else:
        val1 = random.randint(0, 99)
        val2 = random.randint(0, 99)
    return val1, val2

def selectSteppedNumber(first_num):
    val1 = first_num
    val2 = first_num+1
    return val1, val2


def createRandomData(num, is_hard=True, mode="add"):
    out = []
    for i in range(num):
        val1, val2 = selectRandomNumber(is_hard=is_hard)

        if mode == "add":
            ans = val1 + val2
            fom_text = "{}+{}".format(val1, val2)
        elif mode == "sub":
            ans = val1 - val2
            fom_text = "{}-{}".format(val1, val2)
        ans_text = str(ans)
        out.append(fom_text + "," + ans_text + "\n")
    return out


def createSteppedData(num, is_hard=True):
    out = []
    for i in range(num):
        val1, val2 = selectSteppedNumber(i)

        ans = val1 + val2
        fom_text = "{}+{}".format(val1, val2)
        ans_text = str(ans)
        out.append(fom_text + "," + ans_text + "\n")

        if is_hard:
            fom_text = "{}+{}".format(val2, val1)
            out.append(fom_text + "," + ans_text + "\n")
    return out


def createLossData(num, is_hard=True):
    out = []
    # correct 
    for i in range(int(num / 3)):
        val1, val2 = selectRandomNumber(is_hard=is_hard)
        ans = str(val1 + val2)

        fom = "{}+{}".format(val1, val2)
        fom_len = len(fom)
        fom = BEGIN_SYMBOL + fom + END_SYMBOL
        if 5-fom_len > 0:
            for i in range(5-fom_len):
                fom += PAD_SYMBOL

        ans_len = len(ans)
        ans = BEGIN_SYMBOL + ans + END_SYMBOL
        if 3-ans_len > 0:
            for i in range(3-ans_len):
                ans += PAD_SYMBOL

        fom_text = "{},{}".format(fom, ans)
        out.append(fom_text + ",0\n")
    
    # false
    for i in range(int(num / 6)):
        val1, val2 = selectRandomNumber(is_hard=is_hard)
        ans = str(random.randint(0, 198))

        fom = "{}+{}".format(val1, val2)
        fom_len = len(fom)
        fom = BEGIN_SYMBOL + fom + END_SYMBOL
        if 5-fom_len > 0:
            for i in range(5-fom_len):
                fom += PAD_SYMBOL

        ans_len = len(ans)
        ans = BEGIN_SYMBOL + ans + END_SYMBOL
        if 3-ans_len > 0:
            for i in range(3-ans_len):
                ans += PAD_SYMBOL

        fom_text = "{},{}".format(fom, ans)
        out.append(fom_text + ",1\n")

    # false
    for i in range(int(num / 6)):
        val1, val2 = selectRandomNumber(is_hard=is_hard)
        ans = str(val1 + val2 + 1)

        fom = "{}+{}".format(val1, val2)
        fom_len = len(fom)
        fom = BEGIN_SYMBOL + fom + END_SYMBOL
        if 5-fom_len > 0:
            for i in range(5-fom_len):
                fom += PAD_SYMBOL

        ans_len = len(ans)
        ans = BEGIN_SYMBOL + ans + END_SYMBOL
        if 3-ans_len > 0:
            for i in range(3-ans_len):
                ans += PAD_SYMBOL

        fom_text = "{},{}".format(fom, ans)
        out.append(fom_text + ",1\n")

    # not correct sequence
    for i in range(int(num / 6)):
        val1, val2 = selectRandomNumber(is_hard=is_hard)
        ans = str(random.randint(0, 198))
        point = str(random.randint(0, 9))

        fom = "{}{}{}".format(val1, point, val2)
        fom_len = len(fom)
        fom = BEGIN_SYMBOL + fom + END_SYMBOL
        if 5-fom_len > 0:
            for i in range(5-fom_len):
                fom += PAD_SYMBOL

        ans_len = len(ans)
        ans = BEGIN_SYMBOL + ans + END_SYMBOL
        if 3-ans_len > 0:
            for i in range(3-ans_len):
                ans += PAD_SYMBOL

        fom_text = "{},{}".format(fom, ans)
        out.append(fom_text + ",2\n")

    for i in range(int(num / 6)):
        val1, val2 = selectRandomNumber(is_hard=is_hard)
        ans = str(random.randint(0, 198))
        point = str(random.randint(0, 9))

        fom = "{}++{}".format(int(val1/10), val2)
        fom_len = len(fom)
        fom = BEGIN_SYMBOL + fom + END_SYMBOL
        if 5-fom_len > 0:
            for i in range(5-fom_len):
                fom += PAD_SYMBOL

        ans_len = len(ans)
        ans = BEGIN_SYMBOL + ans + END_SYMBOL
        if 3-ans_len > 0:
            for i in range(3-ans_len):
                ans += PAD_SYMBOL

        fom_text = "{},{}".format(fom, ans)
        out.append(fom_text + ",2\n")

    return out



def makeFile(file_name, data_func, num, is_hard=True, mode="add"):
    file_path = "data/" + file_name

    _file = open(file_path, "w")
    out = data_func(num, is_hard, mode)
    _file.writelines(out)
    _file.close()



if __name__ == "__main__":
    makeFile("train_set_99.csv", createRandomData, 50000, is_hard=False, mode="sub")
    makeFile("test_set_99.csv", createRandomData, 10000, is_hard=False, mode="sub")



