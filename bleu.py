import math
from utils import *
from nltk import word_tokenize
from nltk import bleu_score

class BLEU:
    def __init__(self, n_gram):
        self.n_gram = n_gram


    def __call__(self, gen_sentences, ans_sentences):
        score = 0
        for i in range(len(gen_sentences)):
            score += self.calc_bleu(gen_sentences[i], ans_sentences[i])

        return score / len(gen_sentences)
    

    def calc_bleu(self, gen_sentence, ans_sentence):
        if type(gen_sentence) == str:
            gen_sentence = gen_sentence.split(" ")
        if type(ans_sentence) == str:
            ans_sentence = ans_sentence.split(" ")

        anses = [ ans_sentence ]

        BLEUscore = bleu_score.sentence_bleu(anses, gen_sentence, weights=(0.5, 0.5))
        return BLEUscore



if __name__ == "__main__":
    pass