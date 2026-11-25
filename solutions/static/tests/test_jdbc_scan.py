from dataclasses import replace
from pathlib import Path

import jpamb
import jvm
from jdbc_scan import Predicate, scan


def method(signature):
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    return bc.getmethod(jpamb.parse_methodid("jpamb.cases.Jdbc." + signature))


def test_selector_uses_query_and_binding_evidence():
    source = method("queryAlternativesMatched:(ZI)V")
    result = scan(source)
    selected, = result.selected
    assert selected.predicate == Predicate(0, "==", 0)
    assert len(selected.query_sites) == len(selected.jdbc_sites) == 1
    assert selected.query_sites != selected.jdbc_sites
    assert result.constants == {0, 1, 2}

    # An unrelated first input must not win by declaration order.
    shifted = [replace(op, index=op.index + 1) if isinstance(op, (jvm.Load, jvm.Store)) else op
               for op in source.opcodes]
    shifted_id = jpamb.parse_methodid("example.Case.choose:(ZZI)V")
    # The first flag guards an unrelated integer assignment before the JDBC code.
    prefix = [jvm.Load(offset=0, type=jvm.Int(), index=0),
              jvm.Ifz(offset=1, condition=jvm.CmpOpr.Eq, target=4),
              jvm.Push(offset=2, type=jvm.Int(), value=99),
              jvm.Store(offset=3, type=jvm.Int(), index=source.max_locals+1)]
    shifted = [replace(op, target=op.target+4) if isinstance(op, (jvm.Ifz, jvm.If, jvm.Goto))
               else op for op in shifted]
    extra = replace(source, id=shifted_id, opcodes=prefix+shifted, max_locals=source.max_locals+2)
    assert [c.predicate.parameter for c in scan(extra, budget=2).selected] == [1]
    assert 99 in scan(extra).constants


def test_integer_guards_and_independent_budget():
    source = method("selectColumnByChoice:(I)V")
    result = scan(source, budget=2)
    assert [c.predicate for c in result.selected] == [Predicate(0, "==", n) for n in (0, 1)]
    assert all(c.query_sites for c in result.selected)
    assert scan(source, budget=0).selected == ()
    assert scan(source, budget=0).constants == result.constants
