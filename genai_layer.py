
# genai_layer.py
import os, re, json
from typing import Dict, List, Tuple, Optional
import sqlparse
import pandas as pd  # used for dtype checks in insight prompt if needed

from openai import OpenAI

# ---------- Guardrails ----------
DANGEROUS = [r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bALTER\b",
             r"\bATTACH\b", r"\bDETACH\b", r"\bVACUUM\b", r"\bPRAGMA\b"]

# Preferred star-schema names; app will intersect with live schema
CANONICAL_TABLES = ["FactSales", "DimCustomer", "DimProduct", "DimDate"]
JOIN_HINTS = (
    "Joins:\n"
    "- FactSales.Customer_ID = DimCustomer.Customer_ID\n"
    "- FactSales.Product_ID  = DimProduct.Product_ID\n"
    "- FactSales.Order_ID    = DimDate.Date_ID\n"
)

# ---------- OpenAI ----------
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=key)

# ---------- Helpers ----------
def build_allowed_list(schema: Dict[str, List[str]]) -> List[str]:
    prefer = [t for t in CANONICAL_TABLES if t in schema]
    return prefer or list(schema.keys())

def only_single_select(sql: str) -> Tuple[bool, str]:
    stmts = [str(s).strip() for s in sqlparse.parse(sql) if str(s).strip()]
    if len(stmts) != 1: return False, "Multiple SQL statements detected."
    s = stmts[0]
    if not re.match(r"^\s*SELECT\b", s, re.I): return False, "Only SELECT allowed."
    for pat in DANGEROUS:
        if re.search(pat, s.upper()): return False, f"Forbidden keyword: {pat}"
    return True, "OK"

def parse_tables(sql: str) -> List[str]:
    s = re.sub(r"[\\\n\r]+", " ", sql)
    toks = re.split(r"\s+", s)
    out = []
    for i, tok in enumerate(toks):
        if tok.lower() in ("from","join") and i+1 < len(toks):
            tab = re.sub(r"[,();]", "", toks[i+1])
            out.append(tab)
    return out

def schema_grounding_check(sql: str, allowed_present: List[str]) -> Tuple[bool, str]:
    tables_in_sql = {t.lower() for t in parse_tables(sql)}
    allowed = {t.lower() for t in allowed_present}
    if tables_in_sql & allowed:
        return True, "OK"
    return False, "No known table referenced."

def canonicalize_tables(sql: str, schema: Dict[str, List[str]]) -> str:
    def repl(names, target):
        nonlocal sql
        for n in names:
            sql = re.sub(rf"\b{n}\b", target, sql, flags=re.I)
    repl(["factsales","fact_sales","sales","facts"], "FactSales")
    repl(["dimproduct","product","products"], "DimProduct")
    repl(["dimcustomer","customer","customers"], "DimCustomer")
    repl(["dimdate","dates","date"], "DimDate")
    return sql

def table_columns(schema: Dict[str, List[str]]) -> Dict[str, set]:
    return {t: {c.lower() for c in cols} for t, cols in schema.items()}

def closest_column(name: str, candidates: set) -> Optional[str]:
    n = name.lower()
    if n in candidates: return n
    for c in candidates:
        if n in c or c in n: return c
    return None

def rewrite_unknown_columns(sql: str, schema: Dict[str, List[str]]) -> str:
    alias_map = {"f":"FactSales", "p":"DimProduct", "c":"DimCustomer", "d":"DimDate"}
    live_cols = table_columns(schema)
    def repl(m):
        ident = m.group(0)
        alias, col = ident.split(".")
        table = alias_map.get(alias.lower())
        if not table or table not in live_cols: return ident
        cand = closest_column(col, live_cols[table])
        return f"{alias}.{cand}" if cand else ident
    return re.sub(r"\b([fpcd])\.[A-Za-z_][A-Za-z0-9_]*", repl, sql)

def fix_bad_aggregates(sql: str) -> str:
    # Remove stray alias inside aggregates
    for fn in ("SUM","COUNT","AVG","MIN","MAX"):
        sql = re.sub(rf"\b{fn}\s*\(\s*f\.\s*([A-Za-z_][A-Za-z0-9_]*)\s+f\s*\)", rf"{fn}(f.\1)", sql, flags=re.I)
        sql = re.sub(rf"\b{fn}\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+f\s*\)", rf"{fn}(\1)", sql, flags=re.I)
    # Fix table name in aggregate
    sql = re.sub(r"\bSUM\s*\(\s*f\.\s*FactSales\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    sql = re.sub(r"\bSUM\s*\(\s*FactSales\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    # Normalize common synonyms to 'Sales'
    sql = re.sub(r"\bSUM\s*\(\s*f\.(revenue|amount|sales_amount|sale)\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    return sql

def extract_sql(text: str) -> str:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("sql"): return str(obj["sql"]).strip()
    except Exception:
        pass
    m = re.search(r"``````", text, re.S | re.I)
    if m: return m.group(1).strip()
    m2 = re.search(r"(SELECT[\s\S]*?)(?:;|$)", text, re.I)
    return m2.group(1).strip() if m2 else ""

# ---------- Prompting ----------
def build_schema_prompt(schema: Dict[str, List[str]], user_q: str,
                        allowed_present: List[str], prev_turn: Optional[Dict[str,str]]=None) -> str:
    rules = (
        "You convert business questions into ONE valid SQLite SELECT.\n"
        f"- Use ONLY these tables/columns: {', '.join(allowed_present)}; never invent names.\n"
        f"- {JOIN_HINTS}"
        "- Prefer aliases: FactSales f, DimProduct p, DimCustomer c, DimDate d.\n"
        "- Quarters: use DimDate.Order_Date with strftime; Q1=1-3, Q2=4-6, Q3=7-9, Q4=10-12.\n"
        "- Add a LIMIT if the user did not ask for full data.\n"
        'Return ONLY JSON: {"sql":"..."}'
    )
    col_map = "Column map:\n" + "\n".join([f"{t}: {', '.join(schema[t])}" for t in allowed_present if t in schema])
    prev = f"Previous Q: {prev_turn.get('q','')}\nPrevious SQL: {prev_turn.get('sql','')}\n\n" if prev_turn else ""
    examples = (
        "Examples:\n"
        "NL: Top 5 products by revenue in Q4 2024\n"
        "SQL: SELECT p.Product_Name, SUM(f.Sales) AS revenue "
        "FROM FactSales f JOIN DimProduct p ON f.Product_ID = p.Product_ID "
        "JOIN DimDate d ON f.Order_ID = d.Date_ID "
        "WHERE CAST(strftime('%m', d.Order_Date) AS INTEGER) BETWEEN 10 AND 12 "
        "GROUP BY p.Product_Name ORDER BY revenue DESC LIMIT 5;"
    )
    return f"{prev}{rules}\n\n{col_map}\n\nSCHEMA:\n" + \
           "\n".join([f"Table `{t}`: columns = {', '.join(schema[t])}" for t in schema]) + \
           f"\n\n{examples}\nUser question: {user_q}\n\nReturn JSON only."

def call_openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"Convert English to one safe SQLite SELECT; return JSON with key 'sql'."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content

# ---------- Fallbacks ----------
FALLBACKS = [
    (r"\btop\s*\d+\s*products\b.*\bsales\b",
     "SELECT p.Product_Name, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimProduct p ON f.Product_ID = p.Product_ID "
     "GROUP BY p.Product_Name ORDER BY total_sales DESC LIMIT {k};"),
    (r"\bpopular\b.*\bcategory\b|\btop\b.*\bcategory\b",
     "SELECT p.Category, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimProduct p ON f.Product_ID=p.Product_ID "
     "GROUP BY p.Category ORDER BY total_sales DESC LIMIT 5;"),
]

def fallback_sql_for(question: str) -> Optional[str]:
    q = question.lower()
    for pat, tmpl in FALLBACKS:
        if re.search(pat, q):
            k = 10
            m = re.search(r"top\s*(\d+)", q)
            if m: k = int(m.group(1))
            return tmpl.format(k=k)
    return None

# ---------- Insights ----------
def build_insight_prompt(user_q: str, sql: str, df, max_rows: int = 10) -> str:
    sample_df = df.head(max_rows)
    cols = list(sample_df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(sample_df[c])]
    context = {}
    if len(numeric_cols) == 1:
        agg_col = numeric_cols[0]
        total = float(sample_df[agg_col].sum())
        shares = [round((float(v)/total*100.0) if total else 0.0, 2) for v in sample_df[agg_col]]
        context = {"aggregate_col": agg_col, "sample_total": total, "row_shares_pct": shares}
    payload = {"columns": cols, "rows": sample_df.to_dict(orient="records"), "context": context}
    return (
        "Use basic English. Write 2–4 sentences based only on the JSON numbers. "
        "If rows are top items, use 'row_shares_pct' to state rough shares.\n\n"
        f"User question: {user_q}\nSQL: {sql}\nPayload JSON:\n{json.dumps(payload, default=str)}\nSummary:"
    )

def llm_summarize(prompt: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Use basic English. Summarize in 2–4 sentences from provided data."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

# ---------- Main NL→SQL pipeline ----------
def nl_to_sql_and_insight(user_q: str, schema: Dict[str, List[str]],
                          allowed_present: List[str], prev_turn=None):
    result = {"sql": None, "df": None, "summary": None, "error": None}

    prompt = build_schema_prompt(schema, user_q, allowed_present, prev_turn)
    raw = call_openai_chat(prompt)
    sql = extract_sql(raw) or fallback_sql_for(user_q)
    if not sql:
        result["error"] = "Could not extract SQL."; return result

    ok, msg = only_single_select(sql)
    if not ok:
        result["error"] = f"Safety check failed: {msg}"; result["sql"] = sql; return result

    # Canonicalize & ground
    sql = canonicalize_tables(sql, schema)
    ok2, _ = schema_grounding_check(sql, allowed_present)
    if not ok2:
        normalized = re.sub(r"[\\\n\r]+", " ", sql)
        repair = (
            "Repair this SQL to use ONLY these tables and exact columns; never invent names.\n"
            f"Allowed tables: {', '.join(allowed_present)}\n"
            "Forbidden: sales, dim_sales, fact, facts\n"
            f"{JOIN_HINTS}"
            "Use aliases f, p, c, d. Return ONLY JSON {\"sql\":\"...\"}.\n"
            f"Original SQL:\n{normalized}"
        )
        raw2 = call_openai_chat(repair)
        sql2 = extract_sql(raw2)
        if sql2: sql = canonicalize_tables(sql2, schema)

    # Final cleanup and normalization
    sql = re.sub(r"[\\\n\r]+", " ", sql).strip()
    sql = re.sub(r"\bFactSales\b(?!\s+[fF]\b)", "FactSales f", sql)
    sql = re.sub(r"\bDimProduct\b(?!\s+[pP]\b)", "DimProduct p", sql)
    sql = re.sub(r"\bDimCustomer\b(?!\s+[cC]\b)", "DimCustomer c", sql)
    sql = re.sub(r"\bDimDate\b(?!\s+[dD]\b)", "DimDate d", sql)
    sql = rewrite_unknown_columns(sql, schema)
    sql = fix_bad_aggregates(sql)  # critical fix for SUM(f.sales f)

    result["sql"] = sql
    return result
