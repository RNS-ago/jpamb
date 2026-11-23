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
