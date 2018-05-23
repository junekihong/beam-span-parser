import collections.abc

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    # lhuang
    __str__ = linearize

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):

    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]

        label = self.label[-1] if len(self.label) > 0 else ""
        tree = InternalTreebankNode(label, children)

        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        crossing = False
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
            # lhuang: crossing
            if child.left < left < child.right < right or \
               left < child.left < right < child.right:
                crossing = True
        return self, crossing

    # lhuang: like enclosing, but can't be identical
    def next_enclosing(self, left, right):
        assert self.left <= left < right <= self.right \
            and (self.left, self.right) != (left, right)
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right and \
               (child.left, child.right) != (left, right):
                return child.next_enclosing(left, right)
        return self.left, self.right

    def enclosing_crosses(self, left, right):
        crossing = 0
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing_crosses(left, right)
            if child.left < left < child.right < right or \
               left < child.left < right < child.right:
                crossing += 1 + child.num_crosses(left, right)
        return self, crossing
                
    def len_crosses(self, left, right):
        crossing = 0
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.len_crosses(left, right)
            if child.left < left < child.right < right or \
               left < child.left < right < child.right:
                crossing += (child.right - child.left) + child.len_crosses(left, right)
        return crossing

    def enclosing_len_crosses(self, left, right):
        crossing = 0
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing_len_crosses(left, right)
            if child.left < left < child.right < right or \
               left < child.left < right < child.right:
                crossing += (child.right - child.left) + child.len_crosses(left, right)
        return self, crossing


    def oracle_label(self, left, right, cross_span=False):
        correct_aug = 0
        incorrect_aug = 1

        enclosing,crossing = self.enclosing(left, right)
        if cross_span and crossing:
            correct_aug = 1

        if enclosing.left == left and enclosing.right == right:
            return enclosing.label, (correct_aug, incorrect_aug)
        return (), (correct_aug, incorrect_aug)

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right)[0].children
            if left < child.left < right
        ] # lhuang: enclosing now returns (tree, crossing)

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    # lhuang
    def __str__(self):
        return "({}:{}/{})".format(self.left, self.word, self.tag)

    # lhuang: word is not a bracket
    def __len__(self):
        return 0

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees

def right_branching(tree):
    def helper(tree, count):
        #print(count, tree.linearize())
        if isinstance(tree, LeafTreebankNode):
            return (count, tree)
        results = []
        for child in tree.children[:-1]:
            subcount, result_tree = helper(child, 0)
            results.append((subcount, result_tree))
        subcount, result_tree = helper(tree.children[-1], count+1)
        results.append((subcount, result_tree))
        return max(results, key=lambda x: x[0])
    finalcount, tree =  helper(tree, 0)
    return finalcount, tree

def left_branching(tree):
    def helper(tree, count):
        #print(count, tree.linearize())
        if isinstance(tree, LeafTreebankNode):
            return (count, tree)
        results = []
        subcount, result_tree = helper(tree.children[0], count+1)
        results.append((subcount, result_tree))
        for child in tree.children[1:]:
            subcount, result_tree = helper(child, 0)
            results.append((subcount, result_tree))
        return max(results, key=lambda x: x[0])
    finalcount, tree = helper(tree, 0)
    return finalcount, tree

def pretty(tree, level=0, marker='  '):
    pad = marker * level
    if isinstance(tree, LeafTreebankNode) or not tree.children:
        leaf_string = '({} {})'.format(tree.tag, tree.word)
        return pad + leaf_string
    else:
        result = pad + "(" + tree.label
        for child in tree.children:
            result += "\n" + pretty(child, level+1)
        result += ")"
        return result


