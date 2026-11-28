"""JDBC adapter for the group24 CombinedDomain and allocation-site Heap.

Models prepared-statement protocol. Database contents, SQL schema/type checks,
ResultSet cursor state and ordinary Java helper calls are not modelled here.
"""
from dataclasses import replace

from abstractions import Interval
from jdbc_domain import JDBCState, MAX_TRACKED_PARAMETERS
from schema import columns_ok, numeric, parameter_types
from static import CallResult, Finding


def arity_of(sql):
    """Count positional markers in the benchmark SQL subset.

    Skip quoted strings/identifiers and comments. Return unknown for H2
    numbered markers or unsupported quoting instead of guessing.
    """
    i, n = 0, 0
    while i < len(sql):
        c = sql[i]
        if sql.startswith('--', i):
            end = sql.find('\n', i + 2)
            i = len(sql) if end < 0 else end + 1
        elif sql.startswith('/*', i):
            end = sql.find('*/', i + 2)
            if end < 0 or '/*' in sql[i+2:end]:
                return None
            i = end + 2
        elif c in "'\"":
            quote = c
            i += 1
            while i < len(sql):
                if sql[i] == quote:
                    if i + 1 < len(sql) and sql[i+1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                return None
        elif c in '$`[' or sql.startswith('//', i) or c == '\\':
            return None
        elif c == '?':
            if i + 1 < len(sql) and sql[i+1].isdigit():
                return None
            n += 1
            i += 1
        else:
            i += 1
    return n


def _allocate(pc, state, payload):
    heap, ref = state.memory.allocate(pc, payload)
    return CallResult(replace(state, memory=heap), ref)


def connection(pc, state, receiver, args):
    yield _allocate(pc, state, frozenset({'connection'}))


def prepare(pc, state, receiver, args):
    sql, = args
    jdbc = JDBCState.prepare_from_strings(sql, arity_of)
    if jdbc == JDBCState.bot():
        return
    if sql.tainted():
        yield Finding(pc, 'possible-sql-injection', 'SQL may contain user input')
    # H2 rejects an unknown column when the statement is prepared.
    texts = sql.possible_strings()
    if texts is None or not all(columns_ok(t) for t in texts):
        yield Finding(pc, 'unknown-column', 'SQL may name a column its table lacks')
        yield 'sql error'
    yield _allocate(pc, state, jdbc)


def _may_reject_text(jdbc, index, value):
    """True when setString may put a non-number into an INT column (or we cannot tell)."""
    texts = value.possible_strings()
    if texts is not None and all(numeric(t) for t in texts):
        return False
    if jdbc.alternatives is None or index.empty or index.max - index.min > MAX_TRACKED_PARAMETERS:
        return True
    for statement in jdbc.alternatives:
        if statement.query.text is None:
            return True
        types = parameter_types(statement.query.text)
        for i in range(index.min, index.max + 1):
            # Out-of-range indices are reported by the binding check itself.
            if 1 <= i <= len(types) and types[i - 1] != 'VARCHAR':
                return True
    return False


def _bind_interval(jdbc, index):
    if index.empty:
        return jdbc.bind_indices([])
    # Enumerate small intervals precisely. For wide intervals, retain all
    # possible tracked indices and a representative invalid index if needed.
    # Unknown/large arities use the conservative unknown-index transfer.
    if index.max - index.min <= MAX_TRACKED_PARAMETERS:
        return jdbc.bind_indices(range(index.min, index.max + 1))
    if jdbc.alternatives is None or any(
        s.query.arity is None or s.query.arity > MAX_TRACKED_PARAMETERS
        for s in jdbc.alternatives
    ):
        return jdbc.bind(None)
    max_arity = max((s.query.arity for s in jdbc.alternatives), default=0)
    indices = list(range(max(1, index.min), min(max_arity, index.max) + 1))
    if index.min <= 0:
        indices.append(0)
    if index.max > max_arity:
        indices.append(max_arity + 1)
    return jdbc.bind_indices(indices)


def _operation(name, return_kind=None):
    def stub(pc, state, receiver, args):
        jdbc = state.memory.read(receiver)
        if not isinstance(jdbc, JDBCState):
            raise NotImplementedError(f'{name} on non-prepared statement')
        transition = (_bind_interval(jdbc, args[0]) if name == 'bind'
                      else getattr(jdbc, name)())
        for error in sorted(transition.errors):
            yield Finding(pc, error, error)
        if transition.errors:
            yield 'sql error'
        for warning in sorted(transition.warnings):
            yield Finding(pc, warning, warning)
        if transition.state == JDBCState.bot():
            return
        # Heap implements strong updates for unique sites and weak updates
        # for aliases that may refer to several objects/summary sites.
        updated = replace(state, memory=state.memory.write(receiver, transition.state))
        if return_kind == 'resultset':
            yield _allocate(pc, updated, frozenset({'resultset'}))
        elif return_kind == 'batch-array':
            # Current benchmark discards the returned update-count array.
            yield _allocate(pc, updated, frozenset({'batch-array'}))
        elif return_kind == 'int':
            yield CallResult(updated, Interval.top())
        elif return_kind == 'bool':
            yield CallResult(updated, Interval(0, 1))
        else:
            yield CallResult(updated)
    return stub


def create_statement(pc, state, receiver, args):
    yield _allocate(pc, state, frozenset({'statement'}))


def direct_query(pc, state, receiver, args):
    sql, = args
    if sql.possible_strings() == frozenset():
        return
    if sql.tainted():
        yield Finding(pc, 'possible-sql-injection', 'SQL may contain user input')
    # SQL validity is not modelled: preserve both possible SQL failure and success.
    yield 'sql error'
    yield _allocate(pc, state, frozenset({'resultset'}))


def opaque_int(pc, state, receiver, args):
    yield CallResult(state, Interval.top())


def opaque_bool(pc, state, receiver, args):
    yield CallResult(state, Interval(0, 1))


def resultset_get_int(pc, state, receiver, args):
    # Cursor position and DB rows are unknown. This is intentionally imprecise.
    yield 'sql error'
    yield CallResult(state, Interval.top())


JDBC_STUBS = {
    ('jpamb/utils/Db', 'withOrdersTable', '()Ljava/sql/Connection;'): connection,
    ('jpamb/utils/Db', 'withUsersTable', '()Ljava/sql/Connection;'): connection,
    ('java/sql/Connection', 'prepareStatement', '(Ljava/lang/String;)Ljava/sql/PreparedStatement;'): prepare,
    ('java/sql/Connection', 'createStatement', '()Ljava/sql/Statement;'): create_statement,
    ('java/sql/Statement', 'executeQuery', '(Ljava/lang/String;)Ljava/sql/ResultSet;'): direct_query,
    ('java/sql/ResultSet', 'next', '()Z'): opaque_bool,
    ('java/sql/ResultSet', 'getInt', '(I)I'): resultset_get_int,
    ('jpamb/utils/Db', 'count', '(Ljava/sql/Connection;Ljava/lang/String;)I'): opaque_int,
    ('jpamb/utils/Db', 'couponOf', '(Ljava/sql/Connection;I)I'): opaque_int,
    ('jpamb/utils/Db', 'couponsMatch', '(Ljava/sql/Connection;[I)Z'): opaque_bool,
}

for name, desc, operation, kind in (
    ('setInt', '(II)V', 'bind', None),
    ('setString', '(ILjava/lang/String;)V', 'bind', None),
    ('clearParameters', '()V', 'clear_parameters', None),
    ('executeUpdate', '()I', 'execute', 'int'),
    ('executeQuery', '()Ljava/sql/ResultSet;', 'execute', 'resultset'),
    ('execute', '()Z', 'execute', 'bool'),
    ('addBatch', '()V', 'add_batch', None),
    ('executeBatch', '()[I', 'execute_batch', 'batch-array'),
    ('clearBatch', '()V', 'clear_batch', None),
    ('close', '()V', 'close', None),
):
    JDBC_STUBS['java/sql/PreparedStatement', name, desc] = _operation(operation, kind)
    if name in {'executeBatch', 'clearBatch', 'close'}:
        JDBC_STUBS['java/sql/Statement', name, desc] = _operation(operation, kind)


def set_string(pc, state, receiver, args, bind=_operation('bind')):
    jdbc = state.memory.read(receiver)
    if isinstance(jdbc, JDBCState) and _may_reject_text(jdbc, args[0], args[1]):
        yield Finding(pc, 'text-in-int-column', 'setString may bind a non-number to an INT column')
        yield 'sql error'
    yield from bind(pc, state, receiver, args)


JDBC_STUBS['java/sql/PreparedStatement', 'setString', '(ILjava/lang/String;)V'] = set_string
