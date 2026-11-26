from pathlib import Path

import jpamb
import jvm
from abstractions import INT_MAX, ArrayRef, IntArray, Interval
from interval_domain import IntervalDomain
from static import AbstractInterpreter, successors


def test_first_guard_and_read_in_proposal_loop():
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    mid = jpamb.parse_methodid("jpamb.cases.Jdbc.insertOrders:([I)V")
    domain = IntervalDomain()
    engine = AbstractInterpreter.initial(bc, mid, None, domain=domain)
    code = bc.getmethod(mid).opcodes
    length_pc = jvm.state.PC(mid, next(i for i, op in enumerate(code) if isinstance(op, jvm.ArrayLength)))
    read_pc = jvm.state.PC(mid, next(i for i, op in enumerate(code) if isinstance(op, jvm.ArrayLoad)))
    # Start at the first length read, after fixture setup and i = 0.
    initial = engine.states[jvm.state.PC(mid, 0)].store(3, Interval.exact(0))
    state = initial.push(initial.load(3), 3).push(initial.load(0), 0)
    null, (guard, state) = successors(bc, length_pc, state, domain)
    assert null == "null pointer"
    assert state.locals[0] == ArrayRef(frozenset({0}), False)
    branches = dict(successors(bc, guard, state, domain))
    assert branches[guard % bc[guard].target].arrays[0].length == Interval.exact(0)
    body = branches[guard + 1]
    assert body.arrays[0].length == Interval(1, INT_MAX)
    before_read = body.push(body.load(0), 0).push(body.load(3), 3)
    (next_pc, result), = successors(bc, read_pc, before_read, domain)
    assert next_pc == read_pc + 1 and result.stack == (Interval.top(),)

    _, out_of_bounds, (_, _) = successors(
        bc, read_pc, initial.push(initial.load(0), 0).push(Interval.exact(0)), domain
    )
    assert out_of_bounds == "out of bounds"


def test_known_array_and_summary_tail():
    array = IntArray.input(jpamb.case.Array(jvm.Int(), (7, 0)))
    assert array.length == Interval.exact(2)
    assert array.read(Interval.exact(0)) == Interval.exact(7)
    assert array.read(Interval(0, 1)) == Interval(0, 7)
    larger = IntArray.input(jpamb.case.Array(jvm.Int(), tuple(range(100))))
    assert larger.read(Interval(31, 40)) == Interval(31, 99)


def test_write_via_alias_and_read_original():
    from static import State

    array = IntArray.input(jpamb.case.Array(jvm.Int(), (7, 0)))
    ref = ArrayRef(frozenset({0}), False)
    state = State((ref, ref), (), arrays=(array,))
    pc = jvm.state.PC(jpamb.parse_methodid("example.Array.write:([I)V"), 0)
    ops = {pc: jvm.ArrayStore(offset=0, type=jvm.Int()),
           pc + 1: jvm.ArrayLoad(offset=1, type=jvm.Int())}
    # alias[1] = 42; read original[1]
    before = state.push(state.load(1), 1).push(Interval.exact(1)).push(Interval.exact(42))
    (next_pc, after), = successors(ops, pc, before, IntervalDomain())
    (_, result), = successors(ops, next_pc, after.push(after.load(0), 0).push(Interval.exact(1)), IntervalDomain())
    assert result.stack == (Interval.exact(42),)
    assert state.arrays[0].cells == (Interval.exact(7), Interval.exact(0))

    invalid = state.push(ref).push(Interval.exact(2)).push(Interval.exact(42))
    assert list(successors(ops, pc, invalid, IntervalDomain())) == ["out of bounds"]


def test_uncertain_writes_and_join():
    array = IntArray.input(jpamb.case.Array(jvm.Int(), (7, 0)))
    changed = array.write(Interval(0, 1), Interval.exact(42))
    assert changed.cells == (Interval(7, 42), Interval(0, 42))
    assert changed == (array.write(Interval.exact(0), Interval.exact(42)) |
                       array.write(Interval.exact(1), Interval.exact(42)))
    unknown = IntArray.input(None)
    updated = unknown.write(Interval.exact(0), Interval.exact(42))
    assert updated.read(Interval.exact(0)) == Interval.exact(42)
    assert (updated | unknown).read(Interval.exact(0)) == Interval.top()
