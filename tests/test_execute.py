from sql_distill.execute import ExecResult, extract_sql, is_ordered, results_match, run_sql


def test_extract_sql_from_fence_think_and_prose():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("<think>hmm</think>Here you go: SELECT a FROM t; done") == "SELECT a FROM t"
    cte = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert extract_sql(cte) == cte


def test_run_sql_is_read_only_and_single_statement(db):
    assert run_sql(db, "SELECT COUNT(*) FROM customers").rows == ((200,),)
    assert not run_sql(db, "DELETE FROM customers").ok
    assert not run_sql(db, "SELECT 1; SELECT 2").ok
    assert "no such table" in run_sql(db, "SELECT * FROM nope").error


def test_ordered_only_at_top_level():
    assert is_ordered("SELECT a FROM t ORDER BY a")
    assert not is_ordered("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")


def _r(*rows):
    return ExecResult(True, tuple(rows))


def test_match_ignores_column_order_and_extra_columns():
    gold = _r(("Anna", 3), ("Ben", 5))
    assert results_match(gold, _r((5, "Ben", 2), (3, "Anna", 1)), ordered=False)
    assert not results_match(gold, _r(("Anna", 3)), ordered=False)


def test_match_respects_order_only_when_ranked():
    gold = _r(("a",), ("b",))
    flipped = _r(("b",), ("a",))
    assert results_match(gold, flipped, ordered=False)
    assert not results_match(gold, flipped, ordered=True)


def test_match_rounds_floats_and_unifies_ints():
    assert results_match(_r((1234.57,)), _r((1234.5678,)), ordered=False)
    assert results_match(_r((5,)), _r((5.0,)), ordered=False)
    assert not results_match(_r((0.12,)), _r((12.0,)), ordered=False)
