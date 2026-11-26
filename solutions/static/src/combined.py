"""Intervals for ints and booleans, Str for strings, StringBuilder on the heap."""
from dataclasses import replace

import jvm
from abstractions import Interval
from interval_domain import IntervalDomain
from static import CallResult
from strings import Str

STRING = jvm.Object(jvm.ClassName("java.lang.String"))
BUILDER = jvm.ClassName("java.lang.StringBuilder")


class CombinedDomain(IntervalDomain):
    @staticmethod
    def join(a, b):
        if type(a) is type(b):
            return a | b
        # Locals start as the empty interval; a dead slot may also hold different
        # types on two paths. The verifier never lets code read such a slot.
        if isinstance(a, Interval) and a.empty:
            return b
        if isinstance(b, Interval) and b.empty:
            return a
        return Interval.bot()

    def widen(self, old, incoming):
        if isinstance(old, Interval) and isinstance(incoming, Interval):
            return super().widen(old, incoming)
        return self.join(old, incoming)

    @staticmethod
    def literal(typ, value):
        if isinstance(value, str):
            return Str.const(value)
        return IntervalDomain.literal(typ, value)

    @staticmethod
    def parameter(typ, value):
        if typ == STRING:
            return Str.input() if value is None else Str.const(value.value)
        return IntervalDomain.parameter(typ, value)

    @staticmethod
    def allocate(pc, classname, state):
        if classname == BUILDER:
            memory, ref = state.memory.allocate(pc, Str.const(""))
            return replace(state, memory=memory), ref
        return IntervalDomain.allocate(pc, classname, state)


def builder_init(pc, state, receiver, args):
    yield CallResult(replace(state, memory=state.memory.write(receiver, args[0])))


def builder_append(pc, state, receiver, args):
    text = state.memory.read(receiver).concat(args[0])
    yield CallResult(replace(state, memory=state.memory.write(receiver, text)), receiver)


def builder_to_string(pc, state, receiver, args):
    yield CallResult(state, state.memory.read(receiver))


def string_equals(pc, state, receiver, args):
    outcomes = receiver.equals(args[0]) if isinstance(args[0], Str) else {True, False}
    if outcomes:
        yield CallResult(state, Interval(int(min(outcomes)), int(max(outcomes))))


STRING_STUBS = {
    ("java/lang/StringBuilder", "<init>", "(Ljava/lang/String;)V"): builder_init,
    ("java/lang/StringBuilder", "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;"): builder_append,
    ("java/lang/StringBuilder", "toString", "()Ljava/lang/String;"): builder_to_string,
    ("java/lang/String", "equals", "(Ljava/lang/Object;)Z"): string_equals,
}
