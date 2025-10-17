# app.py
import os
import pandas as pd
import streamlit as st

from db_layer import connect_db, introspect_schema, run_sql
from genai_layer import (
    build_allowed_list, nl_to_sql_and_insight,
    build_insight_prompt, llm_summarize
)

st.set_page_config(page_title="Retail Data-to-Insight Assistant", layout="wide")

# --- Main Heading ---
st.markdown(
    "<h1 style='text-align: center; color: #4B9CD3;'>🤖 GenAI-powered Assistant</h1>",
    unsafe_allow_html=True
)


st.set_page_config(page_title="Retail Data-to-Insight Assistant", layout="wide")

# --- Early key check ---
if not (os.getenv("OPENAI_API_KEY") or (hasattr(st, "secrets") and st.secrets.get("OPENAI_API_KEY"))):
    st.error("Add OPENAI_API_KEY in .streamlit/secrets.toml or environment.")
    st.stop()

# --- Shared state ---
if "schema" not in st.session_state:
    con = connect_db()
    st.session_state.schema = introspect_schema(con)
    st.session_state.allowed = build_allowed_list(st.session_state.schema)
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content}]
if "turns" not in st.session_state:
    st.session_state.turns = []      # [{q, sql, insight}]
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None  # sidebar click → run immediately

schema = st.session_state.schema
allowed = st.session_state.allowed

# --- Helpers ---
def short_title(text: str, max_words: int = 8) -> str:
    words = [w for w in text.replace("?", "").split() if w.strip()]
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

def format_sql_multiline(sql: str) -> str:
    s = sql.strip().rstrip(";")
    for k in (" FROM ", " JOIN ", " WHERE ", " GROUP BY ", " ORDER BY ", " LIMIT "):
        s = s.replace(k, "\n" + k.strip() + " ")
    return s + ";"

def run_question(q: str):
    # Emit user bubble
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    prev_turn = st.session_state.turns[0] if st.session_state.turns else None
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and insight..."):
            out = nl_to_sql_and_insight(q, schema, allowed, prev_turn)
        if out.get("error"):
            if any(k in out["error"] for k in ("Schema check failed", "No known table referenced")):
                st.info("Adjusted names to match your schema. Here is the attempted SQL.")
            else:
                st.error(out["error"])
            if out.get("sql"):
                st.code(format_sql_multiline(out["sql"]), language="sql")
            st.session_state.messages.append({"role":"assistant","content": out.get("error","")})
            return
        sql = out["sql"]
        st.code(format_sql_multiline(sql), language="sql")
        try:
            con = connect_db()
            df = run_sql(con, sql)
            st.dataframe(df.head(50))
            ip = build_insight_prompt(q, sql, df)
            summary = llm_summarize(ip)
            st.write(summary)
            st.session_state.turns.insert(0, {"q": q, "sql": sql, "insight": summary})
            st.session_state.messages.append({"role":"assistant","content": summary})
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            st.session_state.messages.append({"role":"assistant","content": f"SQL execution failed: {e}"})

# --- Sidebar (only clickable questions; no SQL here) ---
with st.sidebar:
    st.subheader("Conversation History")
    for i, t in enumerate(st.session_state.turns[:16], 1):
        label = short_title(t["q"])
        if st.button(label, key=f"hist_{i}"):
            # Queue the question to be run in this pass (no st.chat_input prefill)
            st.session_state.pending_q = t["q"]
            st.rerun()  # safe jump to handle in main flow
    st.markdown("---")
    if st.button("Delete history"):
        st.session_state.messages.clear()
        st.session_state.turns.clear()
        st.success("History cleared.")

# --- Render chat history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Run pending question from sidebar if present ---
if st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    run_question(q)

# --- Normal chat input (no 'value' kw) ---
user_q = st.chat_input("Ask a question about your Sales DW")
if user_q:
    run_question(user_q)