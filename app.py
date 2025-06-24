import streamlit as st
import psycopg2
from pgvector.psycopg2 import register_vector
import openai
import json

# Get secrets from Streamlit
db_secrets = st.secrets.connections.postgresql
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_conn():
    conn = psycopg2.connect(
        dbname=db_secrets["database"],
        user=db_secrets["username"],
        password=db_secrets["password"],
        host=db_secrets["host"],
        port=db_secrets["port"],
        sslmode="require"
    )
    register_vector(conn)
    return conn

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def log_query(query, answer):
    """Log user queries and bot responses"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_history (query, answer)
                VALUES (%s, %s)
            """, (query, answer))
        conn.commit()

def hybrid_retrieve_pg(query, top_k=5):
    emb = get_embedding(query)
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, document, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM hcmbot_knowledge
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (emb, emb, top_k*3))
            vector_results = cur.fetchall()
            cur.execute("""
                SELECT id, document, metadata, ts_rank(document_tsv, plainto_tsquery(%s)) AS score
                FROM hcmbot_knowledge
                WHERE document_tsv @@ plainto_tsquery(%s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, top_k*3))
            text_results = cur.fetchall()
    results = {row[0]: (row[1], row[2], row[3], 'vector') for row in vector_results}
    for row in text_results:
        if row[0] not in results or row[3] > results[row[0]][2]:
            results[row[0]] = (row[1], row[2], row[3], 'text')
    sorted_results = sorted(results.values(), key=lambda x: x[2], reverse=True)
    return [(doc, meta) for doc, meta, score, _ in sorted_results[:top_k]]

def chat_with_assistant(query, docs, model="gpt-3.5-turbo"):
    context = "\n\n".join(docs)
    prompt = f"""You are a precise assistant that answers questions based strictly on the provided context.
Context:
{context}

Question: {query}
Answer:"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content

def get_frequent_queries(limit=5):
    """Retrieve most common queries"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, COUNT(*) AS frequency
                FROM query_history
                GROUP BY query
                ORDER BY frequency DESC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()

# Streamlit UI
import streamlit as st

# Top bar: Logo and Version
col1, col2 = st.columns([7, 1])
with col1:
    st.image("newegovlogo.png", width=200)
with col2:
    st.markdown('<div style="text-align: right; font-weight: bold; font-size: 18px; color: #888;">v3</div>', unsafe_allow_html=True)

# Three blank lines after logo
st.markdown("<br>", unsafe_allow_html=True)

# Title and description
st.title("HCM Bot")
st.markdown(
    '<div style="font-size:18px; margin-bottom:20px;">'
    'This bot will assist you in any queries related to the workshop! '
    'Please make sure to be as specific as possible and refrain from using abbreviations.'
    '</div>',
    unsafe_allow_html=True
)

# Ask a question (big and bold), no space before input
st.markdown('<div style="font-size:22px; font-weight:bold; margin-bottom:0px;">Ask a question:</div>', unsafe_allow_html=True)
# Larger input box with prompt
query = st.text_input(
    "",
    value=st.session_state.get('prefilled_query', ''),
    key='query_input',
    placeholder="Enter your query here"
)

if st.button("Submit") and query.strip():
    st.write("Query Received:", query)
    results = hybrid_retrieve_pg(query, top_k=5)
    docs = [doc for doc, meta in results]
    
    if openai.api_key:
        try:
            answer = chat_with_assistant(query, docs)
            st.success(f"Assistant's answer: {answer}")
            log_query(query, answer)  # Log successful interaction
        except Exception as e:
            st.error(f"LLM error: {str(e)}")
    else:
        st.info("Set your OPENAI_API_KEY in .streamlit/secrets.toml to enable LLM answers.")

# No extra space before FAQ
st.markdown('<hr style="margin: 30px 0 0 0;">', unsafe_allow_html=True)
st.subheader("FAQ")
frequent_queries = get_frequent_queries(2)
for idx, (question, freq) in enumerate(frequent_queries):
    if st.button(f"{question} ({freq}×)", key=f"faq_{idx}", use_container_width=True):
        st.session_state.prefilled_query = question

# Optional: Make input box visually bigger (Streamlit doesn't natively support a bigger text_input, but you can use CSS)
st.markdown("""
    <style>
        div[data-baseweb="input"] > div {
            font-size: 20px !important;
            height: 60px !important;
        }
    </style>
""", unsafe_allow_html=True)
