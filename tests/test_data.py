import pytest

from sql_distill.dataset import build_splits
from sql_distill.execute import run_sql
from sql_distill.prompts import SYSTEM, chat
from sql_distill.teacher import Decontaminator
from sql_distill.templates import HELDOUT_TEMPLATES, TRAIN_TEMPLATES


@pytest.fixture(scope="module")
def splits(db):
    return build_splits(db)


def test_seed_is_deterministic(db):
    fingerprint = "SELECT COUNT(*), ROUND(SUM(total), 2), (SELECT COUNT(*) FROM order_items) " \
                  "FROM orders"
    assert run_sql(db, fingerprint).rows == ((2000, 346433740.72, 6049),)


def test_template_ids_unique():
    ids = [t.id for t in TRAIN_TEMPLATES + HELDOUT_TEMPLATES]
    assert len(ids) == len(set(ids))


def test_no_test_sql_in_training(splits):
    train_sql = {e.sql for e in splits["train"]}
    for name in ("test_seen", "test_heldout", "test_handwritten"):
        assert not train_sql & {e.sql for e in splits[name]}, name


def test_heldout_templates_never_trained(splits):
    trained = {e.template for e in splits["train"]}
    assert trained.isdisjoint({t.id for t in HELDOUT_TEMPLATES})


def test_every_gold_query_returns_rows(db, splits):
    for name, rows in splits.items():
        for e in rows:
            res = run_sql(db, e.sql)
            assert res.ok and res.rows, (name, e.question)


def test_decontaminator_blocks_near_copies():
    d = Decontaminator([{"question": "How many customers are in Europe?",
                         "sql": "SELECT COUNT(*) FROM customers WHERE region = 'Europe'"}])
    assert not d.clean("how many customers are in europe", "SELECT 1")
    assert not d.clean("Count them", "select count(*)  from customers where region = 'Europe'")
    assert d.clean("Average price per category?", "SELECT category, AVG(price) FROM products")


def test_prompt_shape():
    msgs = chat("q?", "SELECT 1")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM and "CREATE TABLE orders" in msgs[1]["content"]
