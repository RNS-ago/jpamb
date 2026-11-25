import abstractions as ab
from hypothesis import assume, given
from hypothesis import strategies as st
from strings import Hole, Str, normalize

TEXT = st.text(alphabet="ab? ", max_size=3)


def st_template():
    chunk = st.one_of(TEXT, st.builds(Hole, st.booleans()))
    return st.lists(chunk, max_size=4).map(normalize)


def st_str():
    return st.one_of(
        st.just(Str.top()),
        st.frozensets(st_template(), max_size=Str.K).map(Str),
    )


@st.composite
def st_member(draw, s: Str):
    # A concrete string the abstract value stands for. s must not be bottom.
    if s.templates is None:
        return draw(TEXT)
    t = draw(st.sampled_from(sorted(s.templates, key=str)))
    return "".join(c if isinstance(c, str) else draw(TEXT) for c in t)


@given(st_str(), st_str(), st_str())
def test_str_is_poset(a, b, c):
    ab.is_poset(a, b, c)


@given(st_str(), st_str(), st_str())
def test_str_is_lattice(a, b, c):
    ab.is_lattice(a, b, c)


@given(st.sets(TEXT), st_str())
def test_str_is_galoi(a, b):
    ab.is_galoi(a, b)


@given(st.data(), st_str(), st_str())
def test_join_is_sound(data, a, b):
    assume(a != Str.bot())
    assert data.draw(st_member(a)) in a | b
    if a.tainted():
        assert (a | b).tainted()


@given(st.data(), st_str(), st_str())
def test_concat_is_sound(data, a, b):
    assume(a != Str.bot() and b != Str.bot())
    x, y = data.draw(st_member(a)), data.draw(st_member(b))
    assert x + y in a.concat(b)
    if a.tainted() or b.tainted():
        assert a.concat(b).tainted()


@given(st.data(), st_str(), st_str())
def test_equals_is_sound(data, a, b):
    assume(a != Str.bot() and b != Str.bot())
    x, y = data.draw(st_member(a)), data.draw(st_member(b))
    assert (x == y) in a.equals(b)


def test_normalize_merges_neighbours():
    assert normalize(["a", "", "b", Hole(False), Hole(True)]) == ("ab", Hole(True))


def test_query_alternatives_mismatch():
    sql = Str.const("INSERT INTO orders(id, coupon) VALUES (?, ?)") | Str.const(
        "INSERT INTO orders(id) VALUES (?)"
    )
    assert {q.count("?") for q in sql.possible_strings()} == {1, 2}
    assert not sql.tainted()


def test_select_column_by_choice():
    column = Str.const("name") | Str.const("age") | Str.const("email")
    sql = (
        Str.const("SELECT ")
        .concat(column)
        .concat(Str.const(" FROM users WHERE id = ?"))
    )
    assert sql.possible_strings() == {
        "SELECT name FROM users WHERE id = ?",
        "SELECT age FROM users WHERE id = ?",
        "SELECT email FROM users WHERE id = ?",
    }


def test_count_users_unsafe_is_tainted():
    sql = Str.const("SELECT COUNT(*) FROM users WHERE id = ").concat(Str.input())
    assert sql.tainted()
    assert sql.possible_strings() is None
    assert "SELECT COUNT(*) FROM users WHERE id = 1 OR 1=1" in sql


def test_or_query_loop_reaches_top_after_k():
    sb = Str.const("SELECT COUNT(*) FROM users")
    for _ in range(Str.K - 1):
        sb = sb | sb.concat(Str.const(" OR id = ?"))
    assert len(sb.templates) == Str.K
    sb = sb | sb.concat(Str.const(" OR id = ?"))
    assert sb == Str.top()


def test_say_hello_equals():
    assert Str.input().equals(Str.const("hello")) == {True, False}
    assert Str.const("hello").equals(Str.const("hello")) == {True}
