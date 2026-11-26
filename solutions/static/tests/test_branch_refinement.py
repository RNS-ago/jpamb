from pathlib import Path

import jpamb
import jvm
from abstractions import SignSet
from static import AbstractInterpreter, State


def test_guarded_and_unguarded_division():
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    for name, expected in [
        ("Dependent.safeDivByN", {"ok"}),
        ("Simple.divideByN", {"ok", "divide by zero"}),
    ]:
        method = jvm.AbsMethodID.decode(f"jpamb.cases.{name}:(I)I")
        engine = AbstractInterpreter.initial(bc, method, None)
        outcomes = set()
        while engine.worklist:
            outcomes.update(engine.step()[1])
        assert outcomes == expected
        if name.startswith("Dependent"):
            at_division, = (state for pc, state in engine.states.items()
                            if isinstance(bc[pc], jvm.Binary) and bc[pc].operant == jvm.BinaryOpr.Div)
            assert at_division.locals[0] == SignSet.from_sign("-+")
            assert at_division.stack[-1] == at_division.locals[0]


def test_overwrite_and_join_forget_stack_origin():
    unknown = SignSet.top()
    zero = SignSet.from_sign("0")
    state = State((unknown, unknown), ()).push(unknown, 0)
    overwritten = state.store(0, zero)
    assert overwritten.origins == (None,)
    assert overwritten.stack == (unknown,)
    other = State(state.locals, ()).push(unknown, 1)
    assert (state | other).origins == (None,)
    assert state.narrow(0, zero).stack == (zero,)
