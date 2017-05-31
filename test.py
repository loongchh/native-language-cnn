import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLCNN')
    parser.add_argument('--num-eval', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--feature-dir', type=str, default='data/features/speech_transcriptions/ngrams/2',
                        help='directory containing features, including train/dev directories and \
                              pickle file of (dict, rev_dict) mapping indices to feature labels')
    parser.add_argument('--label', type=str, default='data/labels/dev/labels.dev.csv',
                        help='CSV of the dev set labels')
    parser.add_argument('--log-dir', type=str, default='model',
                        help='directory in which model states are saved')
    parser.add_argument('--time-stamp', type=int, default=None,
                        help='time stamp to training run')
    parser.add_argument('--num-epoch', type=int, default=None,
                        help='epoch at which model state is restored')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables')
    args = parser.parse_args()

    # Create log directory + file
    timestamp = strftime("%Y-%m-%d-%H%M%S")
    log_dir = join(args.log_dir, timestamp)
    makedirs(log_dir)

    # Setup logger
    logging.basicConfig(format='%(name)-s: %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%H:%M:%S')
    logger = logging.getLogger("TRAIN")

    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train(args, logger, log_dir)
