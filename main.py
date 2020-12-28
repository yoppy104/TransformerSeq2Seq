from train import *
from utils import *
from evaluate import *
from enum import Enum
from copy import deepcopy

class RunModes(Enum):
    ALL = 0         # gan training
    G_ONLY = 1      # generator training
    D_ONLY = 2      # discriminator training
    LOSS_TRAIN = 3  # loss_net training
    EMB_TRAIN = 4   # auto_encoder training
    TEST = 5        # debug mode


if __name__ == "__main__":
    run_mode = RunModes.G_ONLY     # change running mode

    infomation_class = Infomation()
    
    train = Train(run_mode==RunModes.TEST)

    infomation_class.start_clock()
    if run_mode == RunModes.TEST:       # Debug Mode
        train.pretrain_g(1)
        infomation_class.lap_time()

        train.pretrain_d(1) 
        infomation_class.lap_time()

        train.train(1)
        infomation_class.lap_time()

    elif run_mode == RunModes.G_ONLY:    # only check generator learning
        train.pretrain_g(PRE_G_EPOCHS)
        infomation_class.lap_time()

    elif run_mode == RunModes.D_ONLY:    # only check discriminator learning
        train = SubTrain()
        train.train(PRE_D_EPOCHS)

    elif run_mode == RunModes.ALL:   # Main Running
        if IS_PRETRAIN_G:
            train.pretrain_g(PRE_G_EPOCHS)
        infomation_class.lap_time()

        if IS_PRETRAIN_D:
            train.pretrain_d(PRE_D_EPOCHS)
        infomation_class.lap_time()

        train.train(MAIN_EPOCHS)
        infomation_class.lap_time()

    elif run_mode == RunModes.LOSS_TRAIN:
        sub_train = SubTrain()
        sub_train.train(PRE_D_EPOCHS)

    elif run_mode == RunModes.EMB_TRAIN:
        train.train_auto_encoder(2000)
        infomation_class.lap_time()

    if run_mode != RunModes.TEST and run_mode != RunModes.EMB_TRAIN:
        infomation_class.finish_clock()
        infomation_class.output_infomation()

        output_results()