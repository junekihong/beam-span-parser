import functools

import heapq
import dynet as dy
import numpy as np

import trees
from collections import defaultdict, OrderedDict
from state import State

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


def augment(scores, oracle_index, aug=(0,1)):
    assert isinstance(scores, dy.Expression)
    correct_aug, incorrect_aug = aug

    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape) * incorrect_aug
    increment[oracle_index] = correct_aug
    return scores + dy.inputVector(increment)


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class ChartParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, lstm_outputs=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        if lstm_outputs is None:
            embeddings = []
            for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_embedding = self.word_embeddings[self.word_vocab.index(word)]
                embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
            lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            #non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            #return dy.concatenate([dy.zeros(1), non_empty_label_scores])
            return self.f_label(get_span_encoding(left, right))

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}
            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, right)
                    if is_train:
                        oracle_label, aug = gold.oracle_label(left, right, self.cross_span)
                        oracle_label_index = self.label_vocab.index(oracle_label) if oracle_label is not None else None

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = augment(label_scores, oracle_label_index, aug)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1].value() +
                                chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (
                        children, label_score + left_score + right_score)

            children, score = chart[0, len(sentence)]

            assert len(children) == 1
            return children[0], score

        tree, score = helper(False)

        if is_train:
            oracle_tree, oracle_score = helper(True)
            assert oracle_tree.convert().linearize() == gold.convert().linearize()
            correct = tree.convert().linearize() == gold.convert().linearize()
            loss = dy.zeros(1) if correct else score - oracle_score
            return tree, loss
        else:
            return tree, score


class BeamParser(object):
    
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
            beamsize,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)

        self.dropout = dropout
        self.beamsize = beamsize

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            return self.f_label(get_span_encoding(left, right))

        @functools.lru_cache(maxsize=None)
        def score_span(left, right, force_gold=False):
            length = right - left
            label_scores = get_label_scores(left,right)

            if is_train:
                oracle_label, aug = gold.oracle_label(left, right, self.cross_span)
                oracle_label_index = self.label_vocab.index(oracle_label)       
                label_scores = augment(label_scores, oracle_label_index, aug)
                
            if force_gold:
                label = oracle_label
                label_score = label_scores[oracle_label_index]
            else:
                label_scores_np = label_scores.npvalue()
                argmax_label_index = int(
                    label_scores_np.argmax() if length < len(sentence) else
                    label_scores_np[1:].argmax() + 1)
                label = self.label_vocab.value(argmax_label_index)
                label_score = label_scores[argmax_label_index]

            return label_score, label

            
        def helper(force_gold):
            if force_gold:
                assert is_train
            if not is_train:
                assert gold is None

            n = len(sentence)
            init = State(-1, 0)
            init.prefix = init.inside = dy.zeroes(1)
            beam = [[init]]
            for step in range(1, 2*n):
                last_beam = beam[-1]
                newbeam = {}

                if self.cubepruning:
                    heap = {}
                    gen_dict = {}

                for state in last_beam:
                    i, j = state.i, state.j
                    if force_gold:
                        if should_reduce(state, gold):
                            leftstate = state.leftptrs[0]
                            k = leftstate.i
                            reduce_score, label = score_span(k, j, force_gold)
                            inside = leftstate.inside + state.inside + reduce_score
                            result = State(k, j)
                            result.label = label
                            result.prefix = leftstate.prefix + state.inside + reduce_score
                            result.inside = leftstate.inside + state.inside + reduce_score
                            result.leftptrs = leftstate.leftptrs
                            result.backptrs = [(leftstate, state)]
                        else:
                            shift_score, label = score_span(j, j+1, force_gold)
                            result = State(j, j+1)
                            result.label = label
                            result.inside = shift_score
                            result.prefix = state.prefix + shift_score
                            result.leftptrs = [state]
                        newbeam[result.i, result.j] = result

                    elif self.cubepruning:
                        if j < n: # can shift
                            # put all the shifted states in the heap (left merged)
                            if (j, j+1) in heap:
                                shift_cost = heap[j, j+1].inside
                                heap[j, j+1].leftptrs.append(state)
                                prefix = state.prefix + shift_cost
                                if prefix.value() > heap[j, j+1].prefix.value():
                                    heap[j, j+1].prefix = prefix
                            else:
                                shift_cost, label = score_span(j, j+1, force_gold)
                                shifted = State(j, j+1)
                                shifted.label = label
                                shifted.inside = shift_cost
                                shifted.prefix = state.prefix + shift_cost
                                shifted.leftptrs = [state]
                                heap[j, j+1] = shifted
                        if i > 0:
                            # put the first unique reduced state in the heap
                            generator = iter(state.leftptrs)
                            try:
                                left = next(generator)
                                k = left.i
                                while (k, j) in gen_dict:
                                    left = next(generator)
                                    k = left.i

                                reduce_score, label = score_span(k, j, force_gold)
                                prefix = left.prefix + state.inside + reduce_score
                                inside = left.inside + state.inside + reduce_score
                                reduced = State(k, j)
                                reduced.label = label
                                reduced.prefix = prefix
                                reduced.inside = inside
                                reduced.leftptrs = left.leftptrs
                                reduced.backptrs = [(left, state)]
                                heap[k, j] = reduced
                                gen_dict[k, j] = generator
                            except(StopIteration):
                                pass
                    else:
                        if j < n: # can shift
                            if (j, j+1) in newbeam:
                                newbeam[j, j+1].leftptrs.append(state)
                            else:
                                shift_cost, label = score_span(j, j+1, force_gold)
                                shifted = State(j, j+1)
                                shifted.label = label
                                shifted.inside = shift_cost
                                shifted.prefix = state.prefix + shift_cost
                                shifted.leftptrs = [state]
                                newbeam[j, j+1] = shifted
                        if i > 0: # can reduce
                            for leftstate in state.leftptrs:
                                k = leftstate.i
                                assert i == leftstate.j
                                reduce_score, label = score_span(k, j, force_gold)

                                prefix = leftstate.prefix + state.inside + reduce_score
                                if (k, j) not in newbeam:
                                    inside = leftstate.inside + state.inside + reduce_score
                                    reduced = State(k, j)
                                    reduced.label = label
                                    reduced.prefix = prefix
                                    reduced.inside = inside
                                    reduced.leftptrs = leftstate.leftptrs
                                    reduced.backptrs = [(leftstate, state)]
                                    newbeam[k,j] = reduced
                                elif prefix.value() > newbeam[k,j].prefix.value():
                                    inside = leftstate.inside + state.inside + reduce_score
                                    newbeam[k,j].label = label
                                    newbeam[k,j].prefix = prefix
                                    newbeam[k,j].inside = inside
                                    newbeam[k,j].leftptrs = leftstate.leftptrs
                                    newbeam[k,j].backptrs = [(leftstate, state)]

                if self.cubepruning:
                    heap = [(-x.prefix.value(), x) for x in heap.values()]
                    heapq.heapify(heap)

                    reduce_actions = 0
                    while heap and (self.beamsize is None or reduce_actions < self.beamsize):
                        reduce_actions += 1
                        _, state = heapq.heappop(heap)
                        generator = gen_dict.get(state.sig, None)
                        assert state.sig not in newbeam
                        newbeam[state.sig] = state
                        
                        if generator is not None:
                            parent = state.backptrs[0][1]
                            j = parent.j
                            try:
                                left = next(generator)
                                k = left.i
                                while (k, j) in gen_dict:
                                    left = next(generator)
                                    k = left.i
                                reduce_score, label = score_span(k, j, force_gold)
                                prefix = left.prefix + parent.inside + reduce_score
                                inside = left.inside + parent.inside + reduce_score
                                reduced = State(k, j)
                                reduced.label = label
                                reduced.prefix = prefix
                                reduced.inside = inside
                                reduced.leftptrs = left.leftptrs
                                reduced.backptrs = [(left, parent)]
                                heapq.heappush(heap, (-prefix.value(), reduced))
                                gen_dict[k, j] = generator

                            except(StopIteration):
                                pass

                if self.beamsize is None:
                    newbeam = sorted(newbeam.values(), reverse=True)
                else:
                    newbeam = sorted(newbeam.values(), reverse=True)[:self.beamsize]
                beam.append(newbeam)

            state = beam[-1][0]
            assert finished(state, n)
            assert state.prefix.value() == state.inside.value(), str((state.score.value(), state.inside.value()))
            tree, history = backtrace(state, sentence)
            return (tree[0], state.prefix), beam

        (tree, score), beam = helper(False)

        if is_train:
            (oracle_tree, oracle_score), oracle_beam = helper(True)
            assert len(oracle_beam) == len(beam)
            assert oracle_tree.convert().linearize() == gold.convert().linearize(), "\n"+str(oracle_tree.convert().linearize()) + "\n" + str(gold.convert().linearize())

            loss = None
            for beam,oracle_beam in zip(beam, oracle_beam):
                gold_state  = oracle_beam[0]
                model_state = beam[0]

                gold_score = gold_state.prefix
                model_score = model_state.prefix

                violation = model_score - gold_score
                if violation.value() < 0:
                    violation = dy.zeroes(1)

                if loss is None or violation.value() > loss.value():
                    loss = violation
            return tree, loss
        return tree, score


def should_reduce(state, gold):
    if state.i <= 0:
        return False
    p, q = gold.next_enclosing(state.i, state.j)
    return p < state.i < state.j <= q


def finished(state, n):
    return (state.i, state.j) == (0, n)


def backtrace(state, sentence):
    i, j, label = state.i, state.j, state.label
    if j == i + 1: # shifted
        tag, word = sentence[i]
        tree = trees.LeafParseNode(i, tag, word)
        if state.label:
            tree = trees.InternalParseNode(label, [tree])
        return [tree], [state]
    else: # reduced
        leftsub, rightsub = state.backptrs[0]
        left_trees, left_states = backtrace(leftsub, sentence)
        right_trees, right_states = backtrace(rightsub, sentence)
        children = left_trees + right_trees
        if label:
            children = [trees.InternalParseNode(label, children)]
        return children, left_states + right_states + [state]
        
