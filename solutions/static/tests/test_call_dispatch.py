from dataclasses import replace

import jvm
from jvm.state import PC
from static import CallResult, SignDomain, State, dispatch_call


def test_instance_arguments_and_memory_update():
    method = jvm.AbsMethodID.decode("java.sql.PreparedStatement.setInt:(II)V")
    op = jvm.InvokeInterface(offset=0, method=method, stack_size=3)
    pc = PC(method, 0)
    receiver, index, value = [SignDomain.literal(jvm.Int(), x) for x in (0, 1, -42)]
    original = State((), (receiver, index, value), index)

    def stub(location, state, ref, args):
        assert (location, ref, args) == (pc, receiver, (index, value))
        assert state.stack == ()
        yield CallResult(replace(state, memory=value))

    key = ("java/sql/PreparedStatement", "setInt", "(II)V")
    assert list(dispatch_call(pc, op, original, {key: stub})) == [
        (pc + 1, State((), (), value))
    ]
    assert original.memory == index


def test_static_zero_argument_return_preserves_stack():
    method = jvm.AbsMethodID.decode("example.Factory.create:()I")
    op = jvm.InvokeStatic(offset=0, method=method)
    pc = PC(method, 0)
    value = SignDomain.literal(jvm.Int(), 1)
    state = State((), (value,))

    def stub(location, after, receiver, args):
        assert receiver is None and args == ()
        yield CallResult(after, value)

    assert list(dispatch_call(pc, op, state, {("example/Factory", "create", "()I"): stub})) == [
        (pc + 1, state.push(value))
    ]


def test_heap_aliases_and_summary_updates():
    from static import Heap
    method = jvm.AbsMethodID.decode("example.Factory.create:()I")
    first, second = PC(method, 0), PC(method, 1)
    heap, a = Heap().allocate(first, frozenset({1}))
    heap, b = heap.allocate(second, frozenset({2}))
    updated = heap.write(a, frozenset({3}))
    assert updated.read(a) == {3} and updated.read(b) == {2}
    assert (heap | updated).read(a) == {1, 3}
    uncertain = updated.write(a | b, frozenset({4}))
    assert uncertain.read(a) == {3, 4} and uncertain.read(b) == {2, 4}
    again, alias = updated.allocate(first, frozenset({5}))
    assert alias == a
    assert again.write(alias, frozenset({6})).read(a) == {3, 5, 6}
