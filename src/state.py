
class State(object):
    __slots__ = "i", "j", "prefix", "inside", "backptrs", "leftptrs", "label", "gold", "h"
    
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j
        self.backptrs = None
        self.leftptrs = None
        self.gold = False

    def score(self):
        return self.prefix.value()

    def __lt__(self, other):
        return self.score() < other.score()

    def __le__(self, other):
        return self.score() <= other.score()

    def __gt__(self, other):
        return self.score() > other.score()

    def __ge__(self, other):
        return self.score() >= other.score()

    @property
    def sig(self):
        return (self.i, self.j)

    def __hash__(self):
        return hash((self.i, self.j))

    def __eq__(self, other):
        return (self.i, self.j) == (other.i, other.j)

    def __str__(self):
        return str((self.i, self.j)) + ":({:5.5},{:5.5},[{}])".format(self.prefix.value(), self.inside.value(), len(self.leftptrs) if self.leftptrs is not None else 0)

    def __repr__(self):
        return str(self)

    def shiftmerge(self, other):
        assert len(other.leftptrs) == 1
        self.leftptrs.append(other.leftptrs[0])

    """
    def finished(self):
        return (self.i, self.j) == (0, len(State.sentence))

    def can_shift(self):
        return self.j < len(State.sentence)

    @staticmethod
    def can_reduce(state):
        return state.i > 0

    @staticmethod
    def should_reduce(state, gold):
        p, q = gold.next_enclosing(state.i, state.j)
        return p < state.i < state.j <= q

    def shift(self, costfunct, gold=None):
        shift_cost, label = costfunct(self.j, self.j+1)

        new = State(self.j, self.j+1)
        new.inside = shift_cost

        new.prefix = self.prefix + shift_cost #+ new.inside

        new.leftptrs = [(self, shift_cost)]
        new.backptrs = [(None, label, "shift")]
        return new

    def reduce(self, costfunct, gold=None):
        for left, shift_cost in self.leftptrs:
            new = State(left.i, self.j)            
            reduce_score, label = costfunct(left.i, self.j)

            new.prefix = left.prefix + self.inside + reduce_score
            new.inside = left.inside + self.inside + reduce_score
            
            new.leftptrs = left.leftptrs
            new.backptrs = [((left, self), label, "reduce")]
            yield new
        yield None
    """


