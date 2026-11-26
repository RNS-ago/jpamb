import jvm
import jpamb
from abstractions import INT_MAX, INT_MIN, Interval
from interval_domain import IntervalDomain
from jdbc_scan import scan
from static import AbstractInterpreter


def test_jvm_arithmetic_edges():
    arithmetic = IntervalDomain.arithmetic
    op = jvm.BinaryOpr
    assert list(arithmetic(Interval.exact(INT_MAX), Interval.exact(1), op.Add)) == [Interval.exact(INT_MIN)]
    assert list(arithmetic(Interval.exact(INT_MIN), Interval.exact(-1), op.Div)) == [Interval.exact(INT_MIN)]
    assert list(arithmetic(Interval.exact(-7), Interval.exact(3), op.Rem)) == [Interval.exact(-1)]
    assert list(arithmetic(Interval.exact(2), Interval(0, 1), op.Div)) == ["divide by zero", Interval.exact(2)]
    assert list(arithmetic(Interval.exact(2), Interval.exact(0), op.Div)) == ["divide by zero"]


def test_loop_widens_with_scanned_threshold_and_reaches_fixpoint():
    # for (int i = 0; i < 10000; i++) {}  -- one abstract state per instruction.
    mid = jvm.AbsMethodID.decode("example.Loop.count:()V")
    integer = jvm.Int()
    ops = [
        jvm.Push(offset=0, type=integer, value=0),
        jvm.Store(offset=1, type=integer, index=0),
        jvm.Load(offset=2, type=integer, index=0),
        jvm.Push(offset=3, type=integer, value=10000),
        jvm.If(offset=4, condition=jvm.CmpOpr.Ge, target=7),
        jvm.Incr(offset=5, index=0, amount=1),
        jvm.Goto(offset=6, target=2),
        jvm.Return(offset=7, type=None),
    ]
    method = jvm.Method(mid, ops, 1)
    bc = jpamb.Bytecode(*jpamb.setup(), {mid: method})
    domain = IntervalDomain(scan(method).constants)
    engine = AbstractInterpreter.initial(bc, mid, None, domain=domain)
    for _ in range(50):
        if not engine.worklist:
            break
        engine.step()
    assert not engine.worklist
    assert engine.states[jvm.state.PC(mid, 2)].locals[0] == Interval(0, 10000)
    assert engine.states[jvm.state.PC(mid, 7)].locals[0] == Interval.exact(10000)
