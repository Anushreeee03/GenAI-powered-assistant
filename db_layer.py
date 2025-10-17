# db_layer.py
import os, re, sqlite3
import pandas as pd
from typing import Dict, List

DB_PATH = r"E:\JOB_TRAINING\4-Retail_Assistant(RAG)\db\SalesDW.db"


def connect_db(path: str = DB_PATH):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def introspect_schema(con) -> Dict[str, List[str]]:
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()]
    schema: Dict[str, List[str]] = {}
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')").fetchall()]
        schema[t] = cols
    return schema

def pretty_schema(schema: Dict[str, List[str]]) -> str:
    return "\n".join([f"Table `{t}`: columns = {', '.join(cols)}" for t, cols in schema.items()])

def run_sql(con, sql: str, default_limit: int = 10000) -> pd.DataFrame:
    q = sql.strip().rstrip(";")
    if not re.search(r"\bLIMIT\b", q, re.I):
        q = f"{q} LIMIT {default_limit}"
    return pd.read_sql_query(q, con)
