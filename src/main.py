import argparse
import itertools
import os.path
import time, timeit
import sys

import dynet as dy
import numpy as np

import evaluate
import parse
import trees
import vocabulary
import gc
from collections import defaultdict

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    sys.setrecursionlimit(10000)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))
    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))
    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")
    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))
    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    if os.path.exists(args.model_path_base + ".meta") and \
       os.path.exists(args.model_path_base + ".data"):
        [parser] = dy.load(args.model_path_base, model)
        args.model_path_base = args.model_path_base.split("_dev")[0] + "-cont"
    elif args.parser_type == "beam-parse":
        parser = parse.BeamParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
            args.beamsize,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    parser.cross_span = args.cross_span
    parser.cubepruning = False if args.nocubepruning else True

    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore() > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore()
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore())
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    
    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                _, loss = parser.parse(sentence, tree)
                
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )
            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()


def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    if args.beamsize is not None:
        parser.beamsize = args.beamsize
    if args.log:
        beamsize = ""
        if isinstance(parser, parse.BeamParser):
            parsertype = "beam"
            beamsize = args.beamsize if args.beamsize is not None else parser.beamsize
        elif isinstance(parser, parse.ChartParser):
            parsertype = "chart"
            beamsize = None
        log = open("log/{}_b{}.log".format(parsertype, beamsize), "w")
    parser.cubepruning = False if args.nocubepruning else True

    test_predicted = []
    score_sum = 0.0

    print("Parsing test sentences...")
    start_time = time.time()
    for i, tree in enumerate(test_treebank):
        #sys.stderr.write("{}\r".format(i))

        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]

        if args.log:
            gc.disable()
            before = time.time()

        predicted, score = parser.parse(sentence)
        
        if args.log:
            elapsed = time.time() - before
            log_string = "len {} model {:10.10} time {}\n".format(len(sentence), score.value(), elapsed)
            log.write(log_string)
            log.flush()
            gc.enable()

        test_predicted.append(predicted.convert())
        score_sum += score.value()

    total_elapsed = float(time.time() - start_time)
    test_fscore = evaluate.evalb(test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {} "
        "\ntotal model score {:10.10} "
        "\nspeed: {}/{:5.5} = {:5.5} => {:5.5}".format(
            test_fscore,
            format_elapsed(start_time),
            score_sum,
            len(test_treebank), total_elapsed, float(len(test_treebank)) / total_elapsed, float(total_elapsed) / len(test_treebank)
        )
    )


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)

    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["chart", "beam-parse"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--beamsize", type=int, default=None)
    subparser.add_argument("--cross-span", default=False, action="store_true")
    subparser.add_argument("--nocubepruning", default=False, action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--beamsize", type=int, default=None)
    subparser.add_argument("--log", default=False, action="store_true")
    subparser.add_argument("--nocubepruning", default=False, action="store_true")

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
