import pytest

from sql_distill.db import build_database


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("db") / "sales.db")
