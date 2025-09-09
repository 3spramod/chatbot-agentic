import os
import pandas as pd
from flask import Flask, request, jsonify
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# ================= CONFIG =================
WORKSPACES = {
    "file1": "Inventory_augmented.xlsx",
    "file2": "car_showroom_dummy.xlsx",
    "file3": "computer_Part_report.xlsx",
}
MODEL_NAME = "gpt-4o-mini-2024-07-18"
EMBED_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Flask app
app = Flask(__name__)

# ================= LLM =================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ================= STATE =================
workspaces_state = {}  # {workspace_name: {"df":..., "agent":..., "rag_tool":..., "pandas_tool":..., "memory":...}}

# ================= BUILD RAG INDEX =================
def build_faiss_index(df):
    docs = []

    # Column metadata docs
    for col in df.columns:
        col_data = df[col].dropna()
        col_type = str(df[col].dtype)
        sample_values = (
            col_data.sample(min(5, len(col_data))).astype(str).tolist()
            if not col_data.empty else []
        )
        meta_summary = f"Column: {col}, Type: {col_type}, Sample: {sample_values}"
        docs.append(Document(page_content=meta_summary, metadata={"type": "column", "col_name": col}))

    # Row chunk docs (40 rows per chunk)
    chunk_size = 40
    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]
        text_repr = chunk.to_csv(index=False)
        docs.append(Document(page_content=text_repr, metadata={"type": "rows", "row_range": f"{start}-{end}"}))

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ================= INGEST FUNCTION =================
def ingest_data(workspace: str):
    if workspace not in WORKSPACES:
        raise ValueError(f"Unknown workspace {workspace}")

    file_path = WORKSPACES[workspace]

    # Load file
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")

    # Build FAISS index
    vectorstore = build_faiss_index(df)

    # RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    def rag_func(query: str):
        response = rag_chain.invoke({"query": query})
        if isinstance(response, dict):
            return response.get("result", "No answer found.")
        return str(response)

    rag_tool = Tool(
        name="RAG-Tool",
        func=rag_func,
        description="Use for EXPLANATIONS, meanings of fields, or free-text descriptions. NOT for structured queries."
    )

    # Pandas agent
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-functions",
        allow_dangerous_code=True
    )

    def run_pandas_query(query: str):
        try:
            response = pandas_agent.invoke({"input": query})
            raw_output = response.get("output") if isinstance(response, dict) else response
            if isinstance(raw_output, pd.DataFrame):
                return raw_output.to_json(orient="records")
            return str(raw_output)
        except Exception as e:
            return f"Error in Pandas agent: {e}"

    pandas_tool = Tool(
        name="Pandas-Agent",
        func=run_pandas_query,
        description="Use for structured lookups like aggregations, filtering, counts, unique values."
    )

    tools = [rag_tool, pandas_tool]
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
    )

    # Save in state
    workspaces_state[workspace] = {
        "df": df,
        "agent": agent,
        "rag_tool": rag_tool,
        "pandas_tool": pandas_tool,
        "memory": memory,
    }


# ================= AUTO-ROUTING =================
def auto_route(user_input: str, workspace: str):
    state = workspaces_state[workspace]
    rag_tool = state["rag_tool"]
    pandas_tool = state["pandas_tool"]
    agent = state["agent"]

    structured_keywords = ["sum", "average", "mean", "count", "unique", "filter", "group by", "sort", "max", "min", "total", "aggregate"]

    if any(kw in user_input.lower() for kw in structured_keywords):
        return pandas_tool.func(user_input), "pandas"
    elif "meaning" in user_input.lower() or "describe" in user_input.lower() or "what is" in user_input.lower():
        return rag_tool.func(user_input), "rag"
    else:
        return agent.invoke({"input": user_input}), "agent"


# ================= API ROUTES =================
@app.route("/ingest/<workspace>", methods=["POST"])
def ingest_route(workspace):
    try:
        ingest_data(workspace)
        return jsonify({"message": f"Data ingested successfully for {workspace}."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query/<workspace>", methods=["POST"])
def query_route(workspace):
    if workspace not in workspaces_state:
        return jsonify({"error": f"No data ingested for {workspace}. Call /ingest/{workspace} first."}), 400

    user_input = request.json.get("query")
    if not user_input:
        return jsonify({"error": "query is required"}), 400

    try:
        response, route = auto_route(user_input, workspace)
        if isinstance(response, dict):
            final_answer = response.get("output") or str(response)
        else:
            final_answer = str(response)
        return jsonify({"response": final_answer, "route": route})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


