from schema import columns_ok, numeric, parameter_types


def test_columns_against_db_schema():
    assert columns_ok("SELECT age FROM users WHERE id = ?")
    assert columns_ok("SELECT COUNT(*) FROM users WHERE id = 1 OR 1=1")
    assert not columns_ok("SELECT email FROM users WHERE id = ?")
    assert not columns_ok("SELECT id FROM customers")


def test_parameter_types():
    assert parameter_types("INSERT INTO orders(id, coupon) VALUES (?, ?)") == ["INT", "INT"]
    assert parameter_types("SELECT name FROM users WHERE id = ?") == ["INT"]
    assert parameter_types("SELECT id FROM users WHERE name = ?") == ["VARCHAR"]
    assert parameter_types("SELECT id FROM customers WHERE x = ?") == [None]


def test_numeric_text():
    # Checked against H2 2.5.252 with an INSERT into orders.coupon.
    for ok in ("7", " 7 ", "+7", "-7", "007", "2147483647"):
        assert numeric(ok), ok
    for bad in ("2147483648", "99999999999", "7.0", "1e3", "seven", "", "1 OR 1=1"):
        assert not numeric(bad), bad
