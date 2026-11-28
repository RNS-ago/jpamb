from pathlib import Path

from abstractions import INT_MAX, INT_MIN
from combined import STRING_STUBS, CombinedDomain
from jdbc_scan import Predicate, scan
from jdbc_stubs import JDBC_STUBS
from static import partitions, regions, run_partitions

import jpamb


def test_regions_cover_every_int():
    for operator in ("==", "<", "<="):
        for c in (INT_MIN, -1, 0, 1, INT_MAX):
            parts = [r for r in regions(Predicate(0, operator, c)) if not r.empty]
            for value in (
                v
                for v in {INT_MIN, c - 1, c, c + 1, INT_MAX}
                if INT_MIN <= v <= INT_MAX
            ):
                assert sum(r.may_equal(value) for r in parts) == 1, (operator, c, value)


def analyse(signature, budget):
    root = Path(__file__).resolve().parents[3]
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(root), jpamb.Effect(None))
    methodid = jpamb.parse_methodid("jpamb.cases.Jdbc." + signature)
    found = scan(bc.getmethod(methodid), budget)
    domain = CombinedDomain(found.constants)
    return run_partitions(
        bc, methodid, domain, STRING_STUBS | JDBC_STUBS, partitions(found.selected)
    )


def test_partitions_keep_query_and_binding_correlated():
    merged, _, incomplete = analyse("queryAlternativesMatched:(ZI)V", budget=0)
    assert not incomplete and "sql error" in merged
    split, findings, incomplete = analyse("queryAlternativesMatched:(ZI)V", budget=2)
    assert not incomplete and "sql error" not in split and "ok" in split
    assert not findings


def test_partitions_keep_real_errors():
    outcomes, _, incomplete = analyse("queryAlternativesMismatch:(Z)V", budget=2)
    assert not incomplete and {"ok", "sql error"} <= outcomes
