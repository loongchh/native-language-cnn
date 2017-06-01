from train import *
from types import SimpleNamespace
from copy import deepcopy
import torch.multiprocessing as mp
from os.path import join
from os import makedirs
import pickle
import logging


# Defaults arguments
args = SimpleNamespace()

args.lr = 1e-4
args.seed = 224
args.regularization = 0
args.clip_norm = None
args.dropout = 0.25
args.num_epochs = 10
args.batch_size = 25
args.val_split = 0.0909
args.max_length = 200
args.embed_dim = 500
args.channel = 500
args.feature_dir = 'data/features/speech_transcriptions/ngrams/2'
args.label = 'data/labels/train/labels.train.csv'
args.save_every = 10
args.cuda = 0


def try_model(lr, reg, do, hyper_dir, cuda=0, logger=None):
    save_dir = join(hyper_dir, "lr-{:1.1e}_reg-{:1.1e}_do-{:1.2f}".format(lr, reg, do))
    makedirs(save_dir)

    args0 = deepcopy(args)
    args0.lr = lr
    args0.regularization = reg
    args0.dropout = do
    args0.cuda = cuda

    print("Training starts at CUDA device {:d}".format(cuda))
    (nlcnn_model, train_loss, train_f1, val_f1) = train(args, progbar=(cuda == 0))

    with open(join(hyper_dir, "lr-{:1.1e}_reg-{:1.1e}_do-{:1.2f}_loss_f1.pkl".format(lr, reg, do))) as fpkl:
        pickle.dump((train_loss, train_f1, val_f1), fpkl)

    save_path = join(hyper_dir, "lr-{:1.1e}_reg-{:1.1e}_do-{:1.2f}_model_state.pkl".format(lr, reg, do))
    torch.save(nlcnn_model.state_dict(), save_path)

    if logger:
        logger.info("lr={:1.1e}, reg={:1.1e}, do={:1.2f} ==> loss={:.3f}, train F1={:.2%}, val F1 = {:.2%}".format(
            lr, reg, do, train_loss[-1], train_f1[-1], val_f1[-1]))


if __name__ == '__main__':
    timestamp = strftime("%Y-%m-%d-%H%M%S")
    hyper_dir = join("hyperparameter", timestamp)
    makedirs(hyper_dir)

    logging.basicConfig(filename=join(hyper_dir, timestamp + ".log"),
                        format='[%(asctime)s] {%(pathname)s:%(lineno)3d} %(levelname)6s - %(message)s',
                        level=logging.DEBUG, datefmt='%H:%M:%S')
    logger = logging.getLogger("HYPER")

    learning_rate = [5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    regularization = [1e-6, 5e-6]#, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    dropout = [0, 0.25, 0.5, 0.75]

    for lr in learning_rate:
        for do in dropout:
            processes = []
            for (i, reg) in enumerate(regularization):
                p = mp.Process(target=try_model, args=(lr, reg, do, hyper_dir, i, logger))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
