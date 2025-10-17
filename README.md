# 🧠 Retail Data-to-Insight Assistant (Task 4)

## 📘 Overview
This project is a **GenAI-powered conversational assistant** that converts **natural language business questions** into **SQL queries**, executes them on a **Sales Data Warehouse (SQLite)**, and presents both **tabular results** and **natural-language insights** — all through an intuitive **Streamlit dashboard**.

---

## ⚙️ Tech Stack
- **Frontend:** Streamlit  
- **Backend / Logic:** Python  
- **Database:** SQLite (SalesDW.db)  
- **AI Model:** OpenAI GPT (used via `genai_layer.py`)  
- **Libraries:** pandas, sqlparse, openai  

---

## 🏗️ Architecture Overview
| Layer | File | Description |
|-------|------|--------------|
| **Frontend** | `app.py` | Streamlit-based UI with chat interface, sidebar, and result visualizations |
| **GenAI Layer** | `genai_layer.py` | Handles LLM prompting, SQL generation, validation, and insight summarization |
| **Database Layer** | `db_layer.py` | Connects to SQLite DB, introspects schema, and executes validated SQL queries |

---

## 🧩 Workflow
1. **User Input:** A business user types a natural-language question (e.g., “Top 5 products in Q4”).  
2. **LLM Processing:**  
   - The question is sent to the `genai_layer` with the database schema context.  
   - The model generates a valid SQL query as JSON output.  
3. **Validation:**  
   - The app verifies that the SQL query is `SELECT`-only and uses allowed tables/columns.  
   - If invalid, a repair prompt corrects schema mismatches.  
4. **Execution:**  
   - The SQL query is run via `db_layer.py` on the local `SalesDW.db`.  
5. **Insight Generation:**  
   - The query result sample is summarized by the LLM into 2–4 meaningful sentences.  
6. **Display:**  
   - The SQL, result table, and insight are shown in Streamlit, along with chat history.

---

## 🧱 Database Schema (Star Schema Example)
| Table | Key Columns | Description |
|--------|--------------|-------------|
| **FactSales** | Sale_ID, Product_ID, Customer_ID, Date_ID, Quantity, Sales | Transaction-level facts |
| **DimProduct** | Product_ID, Product_Name, Category, Price | Product details |
| **DimCustomer** | Customer_ID, Customer_Name, Region | Customer information |
| **DimDate** | Date_ID, Order_Date, Year, Quarter, Month | Date dimensions |

---

## 🧰 Features
✅ Converts NL → SQL → Insight  
✅ Schema auto-detection and repair  
✅ Hallucination control and query validation  
✅ Chat-based memory for previous turns  
✅ Clean dashboard UI with sidebar schema & examples  
✅ Summarized insights from generated data  

---

## 🧠 Hallucination & Safety Controls
- Only **single `SELECT` statements** are executed.  
- Rejects all destructive keywords (`DROP`, `UPDATE`, `DELETE`, etc.).  
- SQL is **schema-grounded** — only allowed tables and columns are used.  
- When unknown names appear, the app triggers a **repair prompt** to LLM.  
- All LLM outputs are parsed and re-validated before execution.  

---

## 🚀 Setup Instructions

### 1️⃣ Prerequisites
- Python 3.10 or above  
- SQLite3 (installed by default with Python)  
- OpenAI API key  

### 2️⃣ Install Dependencies
```bash
pip install streamlit openai pandas sqlparse

3️⃣ Add OpenAI Key

Create a file named .streamlit/secrets.toml and add:

OPENAI_API_KEY = "your-api-key-here"


4️⃣ Run the App
streamlit run app.py

💬 Example Queries to Try

Show total sales by category.”
“Which sub-category generated the highest sales?”
“Top 10 products by total sales amount.”
“Show total sales for each region.”
“Which city in California has the highest sales?”

Deliverables included

app.py: Streamlit UI

genai_layer.py: LLM prompt logic + SQL extraction + insight builder

db_layer.py: DB connection + introspection

README.md (this file)

Challenges & mitigation

Hallucinated column/table names: mitigated by schema grounding + repair prompts + canonicalization functions.

Complex SQL correctness: kept model temperature = 0 and added examples, JOIN hints and column lists.

Data leakage / privacy: only sample rows are sent to summarization; avoid sending full dataset to LLM.



👩‍💻 Developed For

Task 4 — GenAI SQL + Insight Generation
Retail Data-to-Insight Assistant Project


Author: Anushree Sathyan
Tools Used: Streamlit, OpenAI GPT, SQLite, Python