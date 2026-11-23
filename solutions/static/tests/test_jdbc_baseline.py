"""Integration fixture for the two INSERT shapes, pending the production domains."""
from pathlib import Path

import pytest

import jpamb
import jvm
from static import AbstractInterpreter, CallResult, Finding, StringValue
from witness import render_junit


class FixtureString(frozenset):
    def possible_strings(self) -> frozenset[str]:
        return self

    def __or__(self, other):
        return FixtureString(super().__or__(other))


class FixtureDomain:
    # Exact literals suffice for these straight-line paths. The integer
    # parameter is unknown and the database read is opaque to this fixture.
    bottom = staticmethod(frozenset)

    @staticmethod
    def literal(typ, value):
        if isinstance(value, str):
            return FixtureString([value])
        return frozenset([value])

    @staticmethod
    def parameter(typ, value):
        return frozenset([value.value if value is not None else None])

    @staticmethod
    def compare(left, right, op):
        assert op in (jvm.CmpOpr.Eq, jvm.CmpOpr.Ne)
        if None in left or None in right:
            return {True, False}
        return {(a == b) == (op == jvm.CmpOpr.Eq) for a in left for b in right}


def prepare(pc, state, receiver, args):
    # Only the two fixed benchmark SQL literals are supported by this stub.
    (sql,) = args
    assert isinstance(sql, StringValue)
    queries = sql.possible_strings()
    assert queries is not None  # Both fixture queries are literals.
    yield CallResult(state, frozenset(query.count("?") for query in queries))


def bind(pc, state, receiver, args):
    indices, _ = args
    if any(index > count for count in receiver for index in indices):
        yield Finding(pc, "invalid-parameter-index", "Setter index exceeds query arity")
        yield "sql error"
    if any(index <= count for count in receiver for index in indices):
        yield CallResult(state)


def opaque_result(pc, state, receiver, args):
    yield CallResult(state, frozenset([None]))


def analyse_pair_method(signature):
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    mid = jvm.AbsMethodID.decode("jpamb.cases.Jdbc." + signature)
    stubs = {
        ("jpamb/utils/Db", "withOrdersTable", "()Ljava/sql/Connection;"): opaque_result,
        ("java/sql/Connection", "prepareStatement", "(Ljava/lang/String;)Ljava/sql/PreparedStatement;"): prepare,
        ("java/sql/PreparedStatement", "setInt", "(II)V"): bind,
        ("java/sql/PreparedStatement", "executeUpdate", "()I"): opaque_result,
        ("jpamb/utils/Db", "couponOf", "(Ljava/sql/Connection;I)I"): opaque_result,
    }
    engine = AbstractInterpreter.initial(bc, mid, None, domain=FixtureDomain, stubs=stubs)
    while engine.worklist:
        engine.step()
    return engine.findings


def test_merged_query_alternatives():
    finding, = analyse_pair_method("queryAlternativesMismatch:(Z)V")
    assert finding.kind == "invalid-parameter-index"
    with pytest.raises(ValueError, match="concrete witness"):
        render_junit(finding)
    assert "Jdbc.queryAlternativesMismatch(false)" in render_junit(finding, (False,))

    # The baseline loses the correlation between query selection and the setter.
    possible_false_alarm, = analyse_pair_method("queryAlternativesMatched:(ZI)V")
    assert possible_false_alarm.kind == "invalid-parameter-index"
