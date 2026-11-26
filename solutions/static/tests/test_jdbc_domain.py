from abstractions import is_lattice
from jdbc_domain import Binding, JDBCState, QueryShape, MAX_ALTERNATIVES


def prepared(n):
    return JDBCState.prepare([QueryShape(None, n)])


def only(state):
    (alternative,) = state.alternatives
    return alternative


def test_lattice():
    a = prepared(1)
    b = a.bind(1).state
    c = a | b
    is_lattice(a, b, c)
    assert a | a == a
    assert a | b == b | a
    assert a & b == JDBCState.bot()
    assert (a | b) | c == a | (b | c)
    assert a < c < JDBCState.top()


def test_missing_and_reuse():
    s = prepared(2)
    assert s.execute().errors == {"missing-parameter-binding"}
    s = s.bind(1).state.bind(2).state
    result = s.execute()
    assert not result.errors and not result.warnings
    assert only(result.state).bindings == (Binding.RETAINED, Binding.RETAINED)
    reused = result.state.bind(1).state.execute()
    assert reused.warnings == {"retained-parameter"}
    assert not reused.errors


def test_clear_parameters():
    s = prepared(1).bind(1).state.clear_parameters().state
    assert only(s).bindings == (Binding.UNSET,)
    assert s.execute().state == JDBCState.bot()


def test_query_alternatives_mismatch():
    s = prepared(1) | prepared(2)
    result = s.bind(2)
    assert result.errors == {"invalid-parameter-index"}
    assert only(result.state).query.arity == 2
    assert prepared(1).bind(2).state == JDBCState.bot()
    assert prepared(2).bind(0).state == JDBCState.bot()


def test_unknown_index_and_unknown_query():
    result = prepared(2).bind(None)
    assert result.errors == {"invalid-parameter-index"}
    assert len(result.state.alternatives) == 2
    result = JDBCState.prepare(None).bind(1)
    assert result.errors == {"invalid-parameter-index"}
    assert result.state != JDBCState.bot()
    assert "missing-parameter-binding" in result.state.execute().errors


def test_branch_merge():
    s = prepared(1)
    joined = s | s.bind(1).state
    result = joined.execute()
    assert result.errors == {"missing-parameter-binding"}
    assert only(result.state).bindings == (Binding.RETAINED,)


def test_batch_snapshot_and_reuse():
    s = prepared(1).bind(1).state.add_batch().state
    assert (Binding.FRESH,) in only(s).batch
    assert only(s).bindings == (Binding.RETAINED,)
    r = s.add_batch()
    assert "retained-parameter" in r.warnings
    cleared = r.state.clear_parameters().state
    assert only(cleared).bindings == (Binding.UNSET,)
    result = cleared.execute_batch()
    assert not result.errors
    assert "retained-parameter" in result.warnings
    assert only(result.state).batch == frozenset()


def test_incomplete_batch_and_clear_batch():
    r = prepared(1).add_batch()
    assert "missing-parameter-binding" in r.errors
    assert r.state == JDBCState.bot()  # H2 rejects incomplete rows at addBatch.
    s = prepared(1).bind(1).state.add_batch().state.clear_batch().state
    assert not s.execute_batch().errors
    assert only(s).batch == frozenset()


def test_closed_and_close_idempotent():
    s = prepared(1).close().state
    assert s.close().state == s
    assert s.bind(1).errors == {"closed-statement"}
    assert s.execute().state == JDBCState.bot()


def test_bounded_join():
    s = JDBCState.bot()
    for n in range(MAX_ALTERNATIVES + 1):
        s = s | JDBCState.prepare([QueryShape(str(n), 1)])
    assert s == JDBCState.top()
    assert s.bind(1).state == JDBCState.top()


def test_bottom_has_no_successor_or_error():
    assert JDBCState.prepare([]) == JDBCState.bot()
    r = JDBCState.bot().execute()
    assert r.state == JDBCState.bot() and not r.errors and not r.warnings



def test_batch_execution_restores_last_row():
    s = prepared(2).bind(1).state.bind(2).state.add_batch().state
    s = s.bind(1).state.add_batch().state.clear_parameters().state
    assert only(s).bindings == (Binding.UNSET, Binding.UNSET)
    result = s.execute_batch()
    assert not result.errors
    assert only(result.state).bindings == (Binding.RETAINED, Binding.RETAINED)
    assert only(result.state).last_batch is None


def test_empty_batch_does_not_change_bindings():
    s = prepared(2).bind(1).state
    assert s.execute_batch().state == s


def test_clear_batch_preserves_current_bindings():
    s = prepared(1).bind(1).state.add_batch().state.bind(1).state
    cleared = s.clear_batch().state
    assert only(cleared).bindings == (Binding.FRESH,)
    assert only(cleared).last_batch is None


def test_finite_index_alternatives():
    s = prepared(2)
    r = s.bind_indices([1, 2])
    assert not r.errors and len(r.state.alternatives) == 2
    r = s.bind_indices([1, 3])
    assert r.errors == {"invalid-parameter-index"}
    assert only(r.state).bindings == (Binding.FRESH, Binding.UNSET)
    r = s.bind_indices([])
    assert r.state == JDBCState.bot() and not r.errors


def test_string_handoff():
    class Strings:
        def __init__(self, texts):
            self.texts = texts
        def possible_strings(self):
            return self.texts
    arities = {"INSERT INTO orders VALUES (?)": 1,
               "INSERT INTO orders VALUES (?, ?)": 2}
    s = JDBCState.prepare_from_strings(Strings(frozenset(arities)), arities.get)
    assert s.bind(2).errors == {"invalid-parameter-index"}
    assert only(s.bind(2).state).query.arity == 2
    assert JDBCState.prepare_from_strings(Strings(None), arities.get) == JDBCState.prepare(None)
    assert JDBCState.prepare_from_strings(Strings(frozenset()), arities.get) == JDBCState.bot()
    # An unrecognised SQL string retains its text but has unknown arity.
    unknown = JDBCState.prepare_from_strings(Strings(frozenset({"unknown"})), arities.get)
    assert only(unknown).query.arity is None


def test_proposal_coupon_cases():
    for coupons, expected_error, expected_warning in [
        ([0], True, False), ([7, 0], False, True), ([7, 7], False, False)
    ]:
        s = prepared(2)
        errors, warnings = set(), set()
        for coupon in coupons:
            s = s.bind(1).state
            if coupon:
                s = s.bind(2).state
            r = s.add_batch()
            errors.update(r.errors)
            warnings.update(r.warnings)
            s = r.state
        r = s.execute_batch()
        errors.update(r.errors)
        warnings.update(r.warnings)
        assert ("missing-parameter-binding" in errors) == expected_error
        assert ("retained-parameter" in warnings) == expected_warning


def test_batch_loop_fixed_point():
    start = prepared(2)
    state = start
    for _ in range(MAX_ALTERNATIVES + 2):
        next_state = state | state.bind(1).state.bind(2).state.add_batch().state
        if next_state == state:
            break
        state = next_state
    else:
        raise AssertionError("No fixed point")
    assert state != JDBCState.top()


def test_zero_and_large_arity():
    from jdbc_domain import MAX_TRACKED_PARAMETERS
    s = prepared(0)
    assert not s.execute().errors
    assert s.bind(None).state == JDBCState.bot()
    s = prepared(MAX_TRACKED_PARAMETERS + 1)
    assert only(s).bindings is None
    assert s.bind(MAX_TRACKED_PARAMETERS + 2).state == JDBCState.bot()
    assert "missing-parameter-binding" in s.execute().errors


def test_structural_meet_is_not_concrete_refinement():
    unknown = JDBCState.prepare(None)
    concrete = prepared(1)
    # Summary tokens may overlap concretely. Meet is only intersection of tokens.
    assert unknown & concrete == JDBCState.bot()
    assert unknown | concrete != JDBCState.bot()


def test_lattice_and_transfer_monotonicity_sample():
    from itertools import product
    states = [JDBCState.bot(), JDBCState.top(), prepared(0), prepared(1),
              prepared(1).bind(1).state, prepared(1).close().state,
              prepared(1).bind(1).state.add_batch().state, JDBCState.prepare(None)]
    states += [a | b for a, b in product(states, repeat=2)]
    states = list(set(states))
    ops = [lambda s: s.bind(1), lambda s: s.bind(None), lambda s: s.bind_indices([1, 2]),
           JDBCState.clear_parameters, JDBCState.execute, JDBCState.add_batch,
           JDBCState.execute_batch, JDBCState.clear_batch, JDBCState.close]
    for a, b in product(states, repeat=2):
        assert a | b == b | a
        assert a & b == b & a
        assert a | (a & b) == a
        assert a & (a | b) == a
        if a <= b:
            for op in ops:
                x, y = op(a), op(b)
                assert x.state <= y.state
                assert x.errors <= y.errors
                assert x.warnings <= y.warnings



def test_unknown_batch_checked_at_enqueue():
    r = JDBCState.prepare(None).add_batch()
    assert "missing-parameter-binding" in r.errors
    # The normal successor contains only successfully queued H2 rows.
    assert not r.state.execute_batch().errors


def test_batch_overflow_keeps_last_row_and_completeness():
    from jdbc_domain import MAX_BATCH_SHAPES
    s = prepared(5)
    for i in range(1, 6):
        s = s.bind(i).state
    s = s.add_batch().state
    for mask in range(MAX_BATCH_SHAPES + 1):
        for i in range(5):
            if mask & (1 << i):
                s = s.bind(i + 1).state
        s = s.add_batch().state
    assert only(s).batch is None
    s = s.clear_parameters().state
    r = s.execute_batch()
    assert not r.errors
    assert only(r.state).bindings == (Binding.RETAINED,) * 5
    assert only(r.state).batch == frozenset()
