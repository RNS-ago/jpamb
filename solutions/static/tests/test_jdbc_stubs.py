"""Unit-level tests against the actual CombinedDomain heap/stub interfaces."""
import jvm
from jvm.state import PC
from abstractions import Interval
from jdbc_domain import Binding
from jdbc_stubs import JDBC_STUBS, arity_of, prepare
from static import CallResult, Finding, State
from strings import Str


def pc(n):
    return PC(jvm.AbsMethodID.decode('example.C.run:()V'), n)


def call(name, desc, state, ref, *args, n=5):
    return list(JDBC_STUBS['java/sql/PreparedStatement', name, desc](pc(n), state, ref, args))


def normal(results):
    result, = [r for r in results if isinstance(r, CallResult)]
    return result


def prepared(sql='INSERT INTO orders VALUES (?, ?)'):
    r = normal(list(prepare(pc(0), State((), ()), None, (Str.const(sql),))))
    return r.state, r.value


def test_arity():
    assert arity_of("SELECT '?' FROM t WHERE id = ? -- ?") == 1
    assert arity_of('SELECT ?1') is None
    assert arity_of("SELECT 'unterminated") is None
    assert arity_of('SELECT /* ? */ ?') == 1


def test_missing_parameter_and_alias():
    s, ref = prepared()
    alias = ref
    s = normal(call('setInt', '(II)V', s, alias, Interval.exact(1), Interval.exact(5))).state
    r = call('executeUpdate', '()I', s, ref)
    assert 'sql error' in r
    assert any(isinstance(x, Finding) and x.kind == 'missing-parameter-binding' for x in r)
    assert not any(isinstance(x, CallResult) for x in r)
    s = normal(call('setInt', '(II)V', s, ref, Interval.exact(2), Interval.exact(7))).state
    s = normal(call('executeUpdate', '()I', s, ref)).state
    assert next(iter(s.memory.read(alias).alternatives)).bindings == (Binding.RETAINED,)*2


def test_query_alternatives_and_interval():
    r = normal(list(prepare(pc(0), State((), ()), None,
                           (Str.abstract(['INSERT INTO t VALUES (?)', 'INSERT INTO t VALUES (?, ?)']),))))
    results = call('setInt', '(II)V', r.state, r.value, Interval.exact(2), Interval.exact(3))
    assert 'sql error' in results
    state = normal(results).state
    assert {s.query.arity for s in state.memory.read(r.value).alternatives} == {2}
    s, ref = prepared()
    results = call('setInt', '(II)V', s, ref, Interval(-100, 100), Interval.exact(0))
    assert 'sql error' in results
    assert len(normal(results).state.memory.read(ref).alternatives) == 2


def test_close():
    s, ref = prepared()
    s = normal(call('close', '()V', s, ref)).state
    r = call('executeUpdate', '()I', s, ref)
    assert 'sql error' in r
    assert any(isinstance(x, Finding) and x.kind == 'closed-statement' for x in r)
    assert not any(isinstance(x, CallResult) for x in r)


def test_tainted_prepare():
    r = list(prepare(pc(0), State((), ()), None, (Str.input(),)))
    assert any(isinstance(x, Finding) and x.kind == 'possible-sql-injection' for x in r)
    assert normal(r).value



def test_batch_snapshot_through_stubs():
    s, ref = prepared()
    for index in (1, 2):
        s = normal(call('setInt', '(II)V', s, ref, Interval.exact(index), Interval.exact(7))).state
    s = normal(call('addBatch', '()V', s, ref)).state
    s = normal(call('clearParameters', '()V', s, ref)).state
    r = call('executeBatch', '()[I', s, ref, n=6)
    assert 'sql error' not in r
    s = normal(r).state
    (statement,) = s.memory.read(ref).alternatives
    assert statement.bindings == (Binding.RETAINED, Binding.RETAINED)
    assert not statement.batch


def test_interpreter_query_alternatives_mismatch():
    from pathlib import Path
    import jpamb
    from combined import CombinedDomain, STRING_STUBS
    from static import AbstractInterpreter

    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    method = jvm.AbsMethodID.decode('jpamb.cases.Jdbc.queryAlternativesMismatch:(Z)V')
    ai = AbstractInterpreter.initial(
        bc, method, None, domain=CombinedDomain(), stubs=STRING_STUBS | JDBC_STUBS,
    )
    outcomes = set()
    for _ in range(1000):
        if not ai.worklist:
            break
        outcomes.update(ai.step()[1])
    assert not ai.worklist, 'Analysis did not reach a fixed point'
    assert {'ok', 'sql error'} <= outcomes
    assert any(f.kind == 'invalid-parameter-index' for f in ai.findings)
