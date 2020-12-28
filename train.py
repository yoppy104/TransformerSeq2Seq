import torch
import torch.autograd as autograd
import torch.nn as nn

import random

from utils import *
from dataset import *
from soft_argmax import *
from evaluate import *

from generator import *
from discriminator import *
from loss_net import *
from loss_g import *
from loss_d import *
from loss_encdec import * 

from bleu import BLEU


class Train:
    def __init__(self, is_test_mode, transform=None):
        self.test_mode = is_test_mode

        # train_dataset
        # self.train_max_length_s, self.train_max_length_t, self.train_transform, self.train_dataset, self.train_data_loader = LoadSentenceData(DATA_SET_PATH)
        # _, _, _, _, self.train_disc_data_loader = LoadSentenceData(DATA_SET_PATH, transform=self.train_transform)
        # self.test_max_length_s, self.test_max_length_t, self.test_transform, self.test_dataset, self.test_data_loader = LoadSentenceData(TEST_SET_PATH, transform=self.train_transform, _shuffle=False)

        self.train_max_length_s, self.train_max_length_t, self.train_transform, self.train_dataset, self.train_data_loader = LoadTranslateData()
        _, _, _, _, self.train_disc_data_loader = LoadTranslateData(transform=self.train_transform)
        self.test_max_length_s, self.test_max_length_t, self.test_transform, self.test_dataset, self.test_data_loader = LoadTranslateData(mode="test", transform=self.train_transform, _shuffle=False)

        self.train_data_num = len(self.train_dataset)
        self.test_data_num = 200

        # calcurate device
        self.device = torch.device("cuda:0")
            
        # num of vocablary
        self.vocab_size = len(self.train_transform.w2i)

        self.emb_vec = LoadEmbVec("data/word2vec/translate_row.vec.pt", self.train_transform, self.vocab_size).to(self.device)

        self.connect_char_tensor = torch.tensor([self.train_transform.w2i[CONNECT_SYMBOL] for i in range(BATCH_SIZE)]).unsqueeze(1).to(self.device)

        # model
        self.encdec_loss = EncDecLoss(self.device)
        self.bce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.generator = AttentionGenerator(self.train_transform, self.vocab_size, self.device, self.train_max_length_s, self.train_max_length_t, self.bce_loss, self.emb_vec, False)
        # self.discriminator = CNNDiscriminator(EMBEDDING_DIM, self.vocab_size, 0, self.device, gpu=True).to(self.device)
        # self.discriminator = LSTMDiscriminator(EMBEDDING_DIM, HIDDEN_DIM, self.vocab_size, self.train_max_length_s+self.train_max_length_t, 0, self.device)
        self.discriminator = EncDecDiscriminator(EMBEDDING_DIM, HIDDEN_DIM, self.vocab_size, self.train_max_length_s+self.train_max_length_t, 0, self.device)

        self.blue = BLEU(4)

        # initialize param of model
        if IS_LOAD_GEN_MODEL:
            self.generator.load_state_dict(torch.load(LOAD_GEN_MODEL_PATH))
        else:
            self.generator.init_params()
        if IS_LOAD_DISC_MODEL:
            self.discriminator.load_state_dict(torch.load(LOAD_DISC_MODEL_PATH))
        else:
            self.discriminator.init_params()

        # loss function
        self.criterion_g = GLoss()
        # self.criterion_d = DLoss()
        self.criterion_d = GANLoss()

        self.criterion_net = nn.CrossEntropyLoss()

        self.loss_net = LossNet(EMBEDDING_DIM, HIDDEN_DIM, self.vocab_size, 0, self.device).to(self.device)
        if IS_LOAD_NET_MODEL:
            self.loss_net.load_state_dict(torch.load(LOAD_NET_MODEL_PATH))
            for param in self.loss_net.parameters():
                param.requires_grad = False
        else:
            self.loss_net.init_params()


        if IS_LOAD_EMB_MODEL:
            self.generator.encoder.embeddings.load_state_dict(torch.load(LOAD_EMB_MODEL_PATH))
            self.generator.decoder.embeddings.load_state_dict(torch.load(LOAD_EMB_MODEL_PATH))
            self.discriminator.enc_embeddings.load_state_dict(torch.load(LOAD_EMB_MODEL_PATH))
            self.discriminator.dec_embeddings.load_state_dict(torch.load(LOAD_EMB_MODEL_PATH))
            for param in self.generator.encoder.embeddings.parameters():
                param.requires_grad = False
            for param in self.generator.decoder.embeddings.parameters():
                param.requires_grad = False
            for param in self.discriminator.enc_embeddings.parameters():
                param.requires_grad = False
            for param in self.discriminator.dec_embeddings.parameters():
                param.requires_grad = False

        

        # optimizer
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=START_LEARNING_RATE_G, betas=(0.5, 0.999))
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=START_LEARNING_RATE_D, betas=(0.5, 0.999))
        self.optimizer_net = torch.optim.Adam(self.loss_net.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # self.optimizer_enc = torch.optim.Adam(self.generator.encoder.parameters(), lr=START_LEARNING_RATE_G, betas=(0.5, 0.999))
        # self.optimizer_dec = torch.optim.Adam(self.generator.decoder.parameters(), lr=START_LEARNING_RATE_G, betas=(0.5, 0.999))

        self.one = torch.tensor(1, dtype=torch.float).to(self.device)
        self.mone = (self.one * -1).to(self.device)


    def train_generator(self, input_tensor, target_tensor):
        if IS_TRAIN_D_WHEN_TRAINING_G:
            for param in self.discriminator.parameters():
                param.requires_grad = False

        self.generator.zero_grad()

        # generate fake sentences
        z = self.generate_z()
        outputs, bce = self.generator(input_tensor, target_tensor, z, out_mode="loss_word")

        # discriminator loss
        g_d, _ = self.discriminator(input_tensor, outputs)
        loss = g_d.mean()
        loss.backward(self.mone)

        # binary cross entropy loss
        bce.backward()

        self.optimizer_gen.step()

        if IS_TRAIN_D_WHEN_TRAINING_G:
            for param in self.discriminator.parameters():
                param.requires_grad = True

        out = bce.item() + loss.item()

        # del bce
        # del loss

        return out



    def train_discriminator(self, input_tensor, target_tensor):
        self.discriminator.zero_grad()

        # generate fake sentences
        z = self.generate_z()
        fake_sentences = self.generator(input_tensor, target_tensor, z, out_mode="word_only")

        # concatinate inputs and targets, or fake sentences
        # real_sentences = torch.cat([input_tensor, self.connect_char_tensor, target_tensor[:,1:]], dim=1)
        # fake_sentences = torch.cat([input_tensor, self.connect_char_tensor, outputs], dim=1)

        real_sentences = target_tensor[:,1:]

        # discriminator loss
        s_d, emb_s = self.discriminator(input_tensor, real_sentences)
        g_d, emb_g = self.discriminator(input_tensor, fake_sentences)

        emb_s = emb_s.mean(2).mean(1)
        emb_g = emb_g.mean(2).mean(1)

        loss_s = s_d.mean()
        loss_s.backward(self.mone, retain_graph=True)

        loss_g = g_d.mean()
        loss_g.backward(self.one, retain_graph=True)

        gradients = (s_d - g_d)/(emb_s-emb_g+0.00001) # anti 0 division
        gradient_penalty = ((gradients - 1) ** 2).sqrt().mean()
        
        gradient_penalty.backward()

        self.optimizer_disc.step()

        out = loss_g.item() - loss_s.item() + gradient_penalty.item()
        out2 = gradient_penalty.item()
        
        del loss_g
        del loss_s
        del gradient_penalty

        return out, out2


    def change_lr(self, epoch):
        # change learning rate
        if (LR_MANAGEMENT_TYPE == LRType.SEQUENCE):
            if (epoch in LEARNING_RATE_OF_G_DICT.keys()):
                self.optimizer_gen.param_groups[0]['lr'] = LEARNING_RATE_OF_G_DICT[epoch]
            if (epoch in LEARNING_RATE_OF_D_DICT.keys()):
                self.optimizer_disc.param_groups[0]['lr'] = LEARNING_RATE_OF_D_DICT[epoch]
        elif LR_MANAGEMENT_TYPE == LRType.STEP:
            self.optimizer_gen.param_groups[0]['lr'] -= DICRESE_RATE_OF_G_LR
            self.optimizer_disc.param_groups[0]['lr'] -= DICRESE_RATE_OF_D_LR


    # train method
    def train(self, num_epoch):
        print("main training...")

        losses_g = []
        losses_d = []
        loss_gradient = []
        acc_train = []
        acc_test = []
        bleu_train = []
        bleu_test = []

        count = 0
        count_d = 0

        for epoch in range(1, num_epoch+1):
            print("epoch :", epoch)
            print("[", end="")

            self.change_lr(epoch)

            epoch_loss_d = 0
            epoch_gradient_penalty = 0
            epoch_loss_g = 0

            for data in self.train_data_loader:
                print(">", end="")
                count += 1

                # Discriminator train
                for data_disc in self.train_disc_data_loader:
                    count_d += 1

                    input_tensor = data_disc[0].to(self.device)
                    target_tensor = data_disc[1].to(self.device)
                    
                    d_loss, grad_pena = self.train_discriminator(input_tensor, target_tensor)
                    epoch_loss_d += d_loss
                    epoch_gradient_penalty += grad_pena

                input_tensor = data[0].to(self.device)
                target_tensor = data[1].to(self.device)

                # Generator train
                epoch_loss_g += self.train_generator(input_tensor, target_tensor)
            
            losses_g.append(epoch_loss_g / count)
            losses_d.append(epoch_loss_d / count_d)
            loss_gradient.append(epoch_gradient_penalty / count_d)
            print("]")

            # calcurate accuracy
            if epoch % SAMPLE_INTERVAL == 1:
                sentences = self.sample_train()
                acc_train.append(calc_accuracy(sentences))
                anses = [data[2] for data in sentences]
                gens = [data[1] for data in sentences]
                bleu_train.append(self.blue(gens, anses))

                sentences = self.sample_test()
                acc_test.append(calc_accuracy(sentences))
                anses = [data[2] for data in sentences]
                gens = [data[1] for data in sentences]
                bleu_test.append(self.blue(gens, anses))

                # output graph, loss and accuracy
                output_log_data(LOG_LOSS_G_PATH, losses_g)
                output_log_data(LOG_LOSS_D_PATH, losses_d)
                output_log_data(LOG_GRADIENT_PENALTY_PATH, loss_gradient)
                output_log_data(LOG_ACC_TRAIN_PATH, acc_train)
                output_log_data(LOG_ACC_TEST_PATH, acc_test)
                output_log_data(LOG_BLEU_TRAIN_PATH, bleu_train)
                output_log_data(LOG_BLEU_TEST_PAHT, bleu_test)

                output_sentence(sentences, make_path(SAMPLE_SENTENCE_PATH))

                torch.save(self.generator.state_dict(), make_path(GENERATOR_MODEL_PATH))
                torch.save(self.discriminator.state_dict(), make_path(DISCRIMINATOR_MODEL_PATH))

                print("train acc : {}, test acc : {}".format(acc_train[-1], acc_test[-1]))


        if not self.test_mode:
            sentences = self.sample_train()
            acc_train.append(calc_accuracy(sentences))
            anses = [data[2] for data in sentences]
            gens = [data[1] for data in sentences]
            bleu_train.append(self.blue(gens, anses))

            sentences = self.sample_test()
            acc_test.append(calc_accuracy(sentences))
            anses = [data[2] for data in sentences]
            gens = [data[1] for data in sentences]
            bleu_test.append(self.blue(gens, anses))

            # output graph, loss and accuracy
            output_log_data(LOG_LOSS_G_PATH, losses_g)
            output_log_data(LOG_LOSS_D_PATH, losses_d)
            output_log_data(LOG_GRADIENT_PENALTY_PATH, loss_gradient)
            output_log_data(LOG_ACC_TRAIN_PATH, acc_train)
            output_log_data(LOG_ACC_TEST_PATH, acc_test)
            output_log_data(LOG_BLEU_TRAIN_PATH, bleu_train)
            output_log_data(LOG_BLEU_TEST_PAHT, bleu_test)

            output_sentence(sentences, make_path(SAMPLE_SENTENCE_PATH))
            
        # save generator parameters
        torch.save(self.generator.state_dict(), make_path(GENERATOR_MODEL_PATH))
        torch.save(self.discriminator.state_dict(), make_path(DISCRIMINATOR_MODEL_PATH))


    def generate_sub_sentence(self, target_sentences):
        emb = self.generator.encoder.embeddings
        weight = 0.1

        embed_word = emb(target_sentences[:, 0])
        out = embed_word * weight
        for idx in range(1, self.train_max_length_t):
            weight += 0.1
            embed_word = emb(target_sentences[:, idx])
            out += embed_word * weight

        # out /= torch.max(torch.abs(out))

        return out

    def generate_z(self, is_sample=False):
        z_noise = torch.randn(BATCH_SIZE, 1, EMBEDDING_DIM, device=self.device)
        if is_sample:
            return z_noise
        else:
            z_noise = torch.cat([
                z_noise for i in range(self.train_max_length_t-1)
            ], dim=1)
            return z_noise

    # pretraining generator(encoder-decoder)
    def pretrain_g(self, num_epoch):
        print("pretraining generator...")

        loss_net_results = {"total":[], "0":[], "1":[], "2":[]}
        losses = []
        acc_train = []
        acc_test = []
        bleu_train = []
        bleu_test = []
        loss_info = []

        for epoch in range(1, num_epoch+1):
            print("epoch :", epoch)
            print("[", end="")

            self.change_lr(epoch)

            epoch_loss = 0
            count = 0

            for data in self.train_data_loader:
                print(">", end="")

                self.optimizer_gen.zero_grad()

                input_tensor = data[0].to(self.device)
                target_tensor = data[1].to(self.device)

                z = self.generate_z()

                # Encoder-Decoder
                loss = self.generator(input_tensor, target_tensor, z)
                loss.backward() # BinaryCrossEntropy loss

                # gradient cliping
                self.generator.clip_weight(CLIP_RATE)

                
                # print("encodr:emb", self.generator.encoder.embeddings.weight.grad)
                # for weight in self.generator.encoder.lstm.all_weights:
                #     for w in weight:
                #         print("encodr:lstm", w.grad)
                # print("decoder:emb", self.generator.decoder.embeddings.weight.grad)
                # for weight in self.generator.decoder.lstm.all_weights:
                #     for w in weight:
                #         print("decoder:lstm", w.grad)
                # print("decoder:lstm2out", self.generator.decoder.lstm2out.weight.grad)
                # print()

                self.optimizer_gen.step()
                
                # loss data 
                epoch_loss += loss.item()
                count += 1
            
            losses.append(epoch_loss / count)
            print("]")
            print("[loss : {}]".format(losses[-1]))

            # calcurate accuracy
            if epoch % SAMPLE_INTERVAL == 1:
                sentences = self.sample_train()
                anses = [data[2] for data in sentences]
                gens = [data[1] for data in sentences]
                bleu_train.append(self.blue(gens, anses))
                acc_train.append(calc_accuracy(sentences))

                sentences = self.sample_test()
                anses = [data[2] for data in sentences]
                gens = [data[1] for data in sentences]
                bleu_test.append(self.blue(gens, anses))
                acc_test.append(calc_accuracy(sentences))

                print("[train acc: {}, test acc: {}]".format(acc_train[-1], acc_test[-1]))

                output_log_data(LOG_LOSS_G_PATH, losses)
                output_log_data(LOG_ACC_TRAIN_PATH, acc_train)
                output_log_data(LOG_ACC_TEST_PATH, acc_test)
                output_log_data(LOG_BLEU_TRAIN_PATH, bleu_train)
                output_log_data(LOG_BLEU_TEST_PAHT, bleu_test)

                output_sentence(sentences, make_path(SAMPLE_SENTENCE_PATH))
                
                # save generator parameters
                torch.save(self.generator.state_dict(), make_path(GENERATOR_MODEL_PATH))
                torch.save(self.generator.encoder.embeddings.state_dict(), make_path(EMBEDDING_MODEL_PATH))


        if not self.test_mode:
            sentences = self.sample_train()
            anses = [data[2] for data in sentences]
            gens = [data[1] for data in sentences]
            bleu_train.append(self.blue(gens, anses))
            acc_train.append(calc_accuracy(sentences))

            sentences = self.sample_test()
            anses = [data[2] for data in sentences]
            gens = [data[1] for data in sentences]
            bleu_test.append(self.blue(gens, anses))
            acc_test.append(calc_accuracy(sentences))

            # output graph, loss and accuracy
            output_log_data(LOG_LOSS_G_PATH, losses)
            output_log_data(LOG_ACC_TRAIN_PATH, acc_train)
            output_log_data(LOG_ACC_TEST_PATH, acc_test)
            output_log_data(LOG_BLEU_TRAIN_PATH, bleu_train)
            output_log_data(LOG_BLEU_TEST_PAHT, bleu_test)

            output_sentence(sentences, make_path(SAMPLE_SENTENCE_PATH))
        
        # save generator parameters
        torch.save(self.generator.state_dict(), make_path(GENERATOR_MODEL_PATH))
        torch.save(self.generator.encoder.embeddings.state_dict(), make_path(EMBEDDING_MODEL_PATH))


    def output_loss_info(self, out_sentences, net_out, net_loss, nll_loss):
        f = open(make_path(LOSS_INFO_PATH), "w")
        for i in range(BATCH_SIZE):
            sentence = out_sentences[i]
            decoded_sentence = self.train_transform.decode([d.item() for d in sentence])

            label = torch.argmax(net_out[i]).item()

            net = net_loss.item()
            nll = nll_loss.item()

            write_sentence = "   ".join(["".join(decoded_sentence), str(label), str(net), str(nll), "\n"])
            f.write(write_sentence)
        f.close()


    # pretraining discriminator
    def pretrain_d(self, num_epoch):
        print("pretraining discriminator...")

        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        # criterion = DLoss()
        criterion = GANLoss()
        max_iter = (self.train_data_loader.__len__() / 2) - 1

        losses = []

        for i in range(1, num_epoch+1):
            print(i, "epoch")
            print("[", end="")

            count = 0

            epoch_loss = 0
            for sentence in self.train_data_loader:
                input_tensor = sentence[0].to(self.device)
                target_tensor = sentence[1].to(self.device)

                _, _ = self.train_discriminator(input_tensor, target_tensor)

                print(">", end="")
            print("]")

        print("Done.")
        print()

        if not self.test_mode:
            output_graph(losses, make_path(["pre_train", "disc_loss.png"]))

  
    def train_auto_encoder(self, num_epoch):
        losses = []
        auto_encoder = self.generator.decoder
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=1e-4)
        for epoch in range(1, num_epoch+1):
            print("epoch :", epoch)
            for sentences in self.train_data_loader:
                sentences = sentences[0]
                input_tensor = sentences[:, :-1].to(self.device)
                target_tensor = sentences[:, 1:].to(self.device)

                auto_encoder.zero_grad()

                hidden = auto_encoder.init_hidden(BATCH_SIZE)

                z = torch.zeros(BATCH_SIZE, 1, EMBEDDING_DIM, device=self.device)

                hs = None

                loss = 0
                for idx in range(self.train_max_length_s-1):
                    if (hidden[0].size(2) < HIDDEN_DIM):
                        h = torch.cat([hidden[0], hidden[0]], dim=2)
                        c = torch.cat([hidden[1], hidden[1]], dim=2)
                        hidden = (h, c)
                    out, hidden = auto_encoder(input_tensor[:, idx], z, hidden, hs)
                    out = out.squeeze()

                    loss += criterion(out, target_tensor[:, idx])

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            
            print("loss: ", loss.item())

        # output graph, loss and accuracy
        output_log_data(LOG_LOSS_G_PATH, losses)

        sample = self.sample_auto_encoder(auto_encoder)
        output_single_sentence(sample, make_path(SAMPLE_SENTENCE_PATH))

        torch.save(auto_encoder.embeddings.state_dict(), make_path(EMBEDDING_MODEL_PATH))



    # generate sample sentence from train dataset.
    def sample_train(self):
        print("[train.py/sample_train] generate sample sentences.")
        with torch.no_grad():
            self.generator.eval()
            sentences = []
            count = 0
            for data in self.train_data_loader:
                # match the numbers of generating sentences to the test.
                if count > self.test_data_num:
                    break
                count += BATCH_SIZE

                input_tensor = data[0].to(self.device)
                target_tensor = data[1].to(self.device)

                z = self.generate_z(is_sample=True)
                outputs = self.generator.sample(input_tensor, target_tensor, z)

                # translate from torch.tensor to python list.
                for i in range(BATCH_SIZE):
                    sentences.append(
                        [
                            [iid.item() for iid in input_tensor[i,:]],
                            [gid.item() for gid in outputs[i,:]],
                            [tid.item() for tid in target_tensor[i,:]]
                        ]
                    )

            # decode from id list to char sentence.
            for i in range(len(sentences)):
                sentences[i][0] = " ".join(self.train_transform.decode(sentences[i][0]))
                sentences[i][1] = " ".join(self.train_transform.decode(sentences[i][1]))
                sentences[i][2] = " ".join(self.train_transform.decode(sentences[i][2]))

            self.generator.train()
            return sentences

    def sample_auto_encoder(self, encoder):
        print("[train/sample_autoencoder] generate sample sentence")
        with torch.no_grad():
            self.generator.eval()

            word = torch.tensor(
                [self.test_transform.w2i[BEGIN_SYMBOL] for i in range(BATCH_SIZE)]
            ).to(self.device)
            hidden = encoder.init_hidden(BATCH_SIZE)

            z = torch.zeros(BATCH_SIZE, 1, EMBEDDING_DIM, device=self.device)
            hs = None

            sentences = []

            for i in range(self.test_max_length_s):
                output, hidden = encoder(word, z, hidden, hs)
                word = soft_argmax(output.view(BATCH_SIZE, -1, 1, self.vocab_size, 1), self.device)[:, :, 1].squeeze()
                sentences.append(word)
            
            samples = []
            for i in range(BATCH_SIZE):
                temp = [data[i].item() for data in sentences]
                samples.append(temp)
            
            for i in range(len(samples)):
                samples[i] = " ".join(self.test_transform.decode(samples[i]))
            
            self.generator.train()
            return samples


    # generate sample sentence from test dataset.
    def sample_test(self):
        print("[train.py/sample_test] generate sample sentences.")
        with torch.no_grad():
            self.generator.eval()

            sentences = []
            count = 0
            for data in self.test_data_loader:
                if count > self.test_data_num:
                    break
                count += BATCH_SIZE

                input_tensor = data[0].to(self.device)
                target_tensor = data[1].to(self.device)

                z = self.generate_z(is_sample=True)
                outputs = self.generator.sample(input_tensor, target_tensor, z)

                # translate from torch.tensor to python list.
                for i in range(BATCH_SIZE):
                    sentences.append(
                        [
                            [iid.item() for iid in input_tensor[i,:]],
                            [gid.item() for gid in outputs[i,:]],
                            [tid.item() for tid in target_tensor[i,:]]
                        ]
                    )
            
            # decode from id list to char sentence.
            for i in range(len(sentences)):
                sentences[i][0] = " ".join(self.test_transform.decode(sentences[i][0]))
                sentences[i][1] = " ".join(self.test_transform.decode(sentences[i][1]))
                sentences[i][2] = " ".join(self.test_transform.decode(sentences[i][2]))
            
            self.generator.train()
            return sentences


class SubTrain:
    def __init__(self):
        # train_dataset
        self.train_max_length_s, self.train_max_length_t, self.train_transform, self.train_dataset, self.train_data_loader = LoadLossData(DISC_TRAIN_PATH)
        self.test_max_length_s, self.test_max_length_t, self.test_transform, self.test_dataset, self.test_data_loader = LoadLossData(DISC_TEST_PATH)

        self.train_data_num = len(self.train_dataset)
        self.test_data_num = len(self.test_dataset)

        # calcurate device
        self.device = torch.device("cuda:0")
            
        # num of vocablary
        self.vocab_size = len(self.train_transform.w2i)

        # model
        self.loss_net = LossNet(EMBEDDING_DIM, HIDDEN_DIM, self.vocab_size, 0, self.device).to(self.device)
        self.loss_net.init_params()

        self.criterion = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.loss_net.parameters(), lr=1e-4)

    
    def train(self, num_epoch):
        losses = []
        acc_train = []
        acc_test = []

        train_acc = 0
        epoch = 0

        #for epoch in range(1, num_epoch+1):
        while(epoch < num_epoch):
            epoch += 1
            print("epoch :", epoch)
            epoch_loss = 0

            for data in self.train_data_loader:
                input_tensor = data[0].to(self.device)
                target_tensor = data[1].squeeze().to(self.device)

                self.loss_net.init_hidden(BATCH_SIZE)

                for i in range(self.train_max_length_s):
                    out = self.loss_net(input_tensor[:,i])
                
                out = out.squeeze()

                loss = self.criterion(out, target_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            losses.append(epoch_loss)

            if epoch == 1 or epoch % 100 == 0:
                with torch.no_grad():
                    train_acc = self.evaluate_acc("train")
                    test_acc = self.evaluate_acc("test")

                acc_train.append(train_acc)
                acc_test.append(test_acc)
                print("[loss:{}, train acc:{}, test acc:{}]".format(epoch_loss, train_acc, test_acc))
            else:
                print("[loss:{}]".format(epoch_loss))
    
        # output graph, loss and accuracy
        output_log_data(LOG_LOSS_G_PATH, losses)
        output_log_data(LOG_ACC_TRAIN_PATH, acc_train)
        output_log_data(LOG_ACC_TEST_PATH, acc_test)

        # save model parameters
        torch.save(self.loss_net.state_dict(), make_path(LOSS_NET_MODEL_PATH))

    
    def evaluate_acc(self, mode):
        if mode == "train":
            dataloader = self.train_data_loader
            max_length = self.train_max_length_s
        elif mode == "test":
            dataloader = self.test_data_loader
            max_length = self.test_max_length_s

        acc = 0
        count = 0

        # run only one data
        for data in dataloader:
            input_tensor = data[0].to(self.device)
            target_tensor = data[1].to(self.device)

            self.loss_net.init_hidden(BATCH_SIZE)

            for i in range(max_length):
                out = self.loss_net(input_tensor[:,i])
            
            out = torch.argmax(out, dim=1)

            for j in range(BATCH_SIZE):
                if out[j].item() == target_tensor[j].item():
                    acc += 1
            count += BATCH_SIZE
            break

        acc = acc / count * 100
        
        return acc