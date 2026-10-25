import sql_distill.train as train


def test_execution_reward_grades_correct_wrong_and_invalid(db, monkeypatch):
    monkeypatch.setattr(train, "DEFAULT_DB", db)
    gold = "SELECT COUNT(*) FROM customers WHERE tier = 'enterprise'"
    completions = [
        "```sql\nSELECT count(*) FROM customers WHERE tier='enterprise';\n```",
        "SELECT COUNT(*) FROM customers",
        "SELECT FROM WHERE",
    ]
    assert train.execution_reward(completions, [gold] * 3) == [1.0, 0.1, 0.0]
