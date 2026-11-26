from pathlib import Path

import jpamb
import jvm
from abstractions import Interval
from combined import STRING_STUBS, CombinedDomain
from static import AbstractInterpreter, CallResult
from strings import Str


def run(method, extra_stubs):
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    ai = AbstractInterpreter.initial(
        bc, jvm.AbsMethodID.decode(method), None,
        domain=CombinedDomain(), stubs=STRING_STUBS | extra_stubs,
    )
    outcomes = set()
    while ai.worklist:
        outcomes |= ai.step()[1]
    return outcomes


def fake_jdbc(queries):
    # Opaque references stand in for the JDBC part until its stubs exist.
    def returns(value):
        return lambda pc, state, receiver, args: [CallResult(state, value)]

    def prepare(pc, state, receiver, args):
        queries.append(args[0])
        yield CallResult(state, frozenset({"ps"}))

    return {
        ("jpamb/utils/Db", "withUsersTable", "()Ljava/sql/Connection;"): returns(frozenset({"conn"})),
        ("java/sql/Connection", "prepareStatement", "(Ljava/lang/String;)Ljava/sql/PreparedStatement;"): prepare,
        ("java/sql/PreparedStatement", "setInt", "(II)V"): returns(None),
        ("java/sql/PreparedStatement", "executeQuery", "()Ljava/sql/ResultSet;"): returns(frozenset({"rs"})),
        ("java/sql/ResultSet", "next", "()Z"): returns(Interval(0, 1)),
        ("java/sql/ResultSet", "getInt", "(I)I"): returns(Interval.top()),
    }


def test_builder_keeps_branch_alternatives():
    queries = []
    run("jpamb.cases.Jdbc.selectColumnByChoice:(I)V", fake_jdbc(queries))
    assert queries[-1] == Str.abstract(
        f"SELECT {c} FROM users WHERE id = ?" for c in ("name", "age", "email")
    )


def test_builder_in_loop_terminates_and_is_sound():
    queries = []
    outcomes = run("jpamb.cases.Jdbc.orQuery:([I)V", fake_jdbc(queries))
    assert "SELECT COUNT(*) FROM users WHERE id = ? OR id = ?" in queries[-1]
    # ids[i] sits under the guard i < ids.length, and inputs are never null.
    assert outcomes == {"ok", "assertion error"}


def test_string_input_is_tainted():
    assert CombinedDomain.parameter(jvm.Object(jvm.ClassName("java.lang.String")), None).tainted()
