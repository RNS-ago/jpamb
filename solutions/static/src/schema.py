"""Column and type checks against the fixed tables in jpamb.utils.Db."""
import re

# Mirrors the CREATE TABLE statements in cases/jpamb/utils/Db.java.
SCHEMA = {
    "orders": {"id": "INT", "coupon": "INT"},
    "users": {"id": "INT", "name": "VARCHAR", "age": "INT"},
}
KEYWORDS = {
    "select", "from", "where", "insert", "into", "values", "update", "set", "delete",
    "and", "or", "not", "null", "is", "in", "as", "count", "order", "by", "limit",
}
TOKEN = re.compile(r"'(?:[^']|'')*'|\w+|\S")
NAME = re.compile(r"[A-Za-z_]\w*")


def _tokens(sql):
    return TOKEN.findall(sql)


def _table(tokens):
    tables = {t.lower() for t in tokens if t.lower() in SCHEMA}
    return tables.pop() if len(tables) == 1 else None


def columns_ok(sql: str) -> bool:
    """False when sql may name a column its table lacks, or the table is unknown."""
    tokens = _tokens(sql)
    table = _table(tokens)
    if table is None:
        return False
    names = (t.lower() for t in tokens if NAME.fullmatch(t))
    return all(n in KEYWORDS or n in SCHEMA or n in SCHEMA[table] for n in names)


def parameter_types(sql: str) -> list[str | None]:
    """The column type behind each ? in order, None where it cannot be told."""
    tokens = _tokens(sql)
    table = _table(tokens)
    if table is None:
        return [None] * tokens.count("?")
    columns = SCHEMA[table]
    lower = [t.lower() for t in tokens]
    # INSERT INTO t(c1, c2) VALUES (?, ?): the i-th value goes to the i-th column.
    if lower[:2] == ["insert", "into"] and "values" in lower:
        split = lower.index("values")
        names = [t for t in lower[:split] if t in columns]
        values = [t for t in tokens[split + 1:] if t not in "(),"]
        return [columns.get(names[i]) if i < len(names) else None
                for i, t in enumerate(values) if t == "?"]
    # col = ?, col < ?, ...: the column right before the operator.
    return [columns.get(lower[i - 2]) if i >= 2 and tokens[i - 1] in "=<>" else None
            for i, t in enumerate(tokens) if t == "?"]


def numeric(text: str) -> bool:
    # H2 2.5.252 accepts " 7 ", "+7" and "007" for an INT column, and rejects
    # "7.0", "1e3", "" and anything outside the 32-bit range.
    if re.fullmatch(r"\s*[+-]?\d+\s*", text) is None:
        return False
    return -(2**31) <= int(text) <= 2**31 - 1
