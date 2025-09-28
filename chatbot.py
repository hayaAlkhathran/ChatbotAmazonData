import streamlit as st
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq 

# ===============================
#  API Key
# ===============================

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Data/amazon_product_reviews_updated1.xlsx")
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please check the file path.")
        return pd.DataFrame()
    return df

df = load_data()

# ===============================
# Combine Text Columns
# ===============================
if not df.empty:
    for col in ["features", "Feedback_review"]:
        if col not in df.columns:
            df[col] = ""  

    df["combined_text"] = (
        df["product_name"].astype(str) + " | " +
        df["category"].astype(str) + " | " +
        df["about_product"].astype(str) + " | " +
        df["review_title"].astype(str) + " | " +
        df["review_content"].astype(str) + " | " +
        df["features"].astype(str) + " | " +
        df["Feedback_review"].astype(str)
    )

# ===============================
#  Build Embeddings + FAISS (Local LLaMA)
# ===============================
@st.cache_resource
def build_index(dataframe):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(dataframe["combined_text"].tolist(), convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return embedder, index

if not df.empty:
    embedder, index = build_index(df)

# ===============================
# Load Local LLaMA Model
# ===============================
@st.cache_resource
def load_model():
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=True)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=True
        )
    else:
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=True
        )

    return tokenizer, model
tokenizer, model = load_model()

# ===============================
# RAG Retrieval (Local LLaMA)
# ===============================
def retrieve_context(query, top_k=3):
    if df.empty:
        return "No dataset loaded."
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]]
    return results

def handle_structured_query(query):
    q = query.lower()
    if "how many category" in q or "number of categories" in q:
        n = df["category"].nunique()
        return f"The dataset contains {n} unique categories."
    if "top" in q and "products by rating" in q:
        top_products = df.sort_values("rating", ascending=False).head(5)[["product_name", "rating"]]
        return "Here are the top 5 products by rating:\n" + top_products.to_string(index=False)
    return None

def rag_answer(user_query):
    structured = handle_structured_query(user_query)
    if structured:
        return structured

    context_df = retrieve_context(user_query, top_k=3)
    if isinstance(context_df, str):
        context_text = context_df
    else:
        context_text = context_df[
            ["product_name", "category", "discounted_price", "actual_price",
             "rating", "features", "Feedback_review", "review_title", "review_content"]
        ].to_string(index=False)

    prompt = f"""
    You are a helpful AI assistant.
    User question: {user_query}

    Context from the dataset:
    {context_text}

    Answer politely and clearly. If the dataset has relevant info, use it.
    If not, reply generally.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ===============================
# Groq API Chatbot (no embeddings)
# ===============================
groq_client = Groq(api_key=GROQ_API_KEY)

def groq_chat_answer(user_query, df):
    if df.empty:
        context_text = "No dataset is loaded."
    else:
        
        context_text = "Columns: " + ", ".join(df.columns) + "\n"
        context_text += "Example row:\n" + df.iloc[0].to_string()

    prompt = f"""
    You are a helpful assistant. 
    The user will ask questions about the following dataset.

    Dataset sample:
    {context_text}

    User question: {user_query}

    Answer clearly using only the dataset context above.
    If the answer is not available, say 'The dataset does not contain this information.'
    """

    chat_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return chat_completion.choices[0].message.content

# ===============================
# Streamlit UI
# ===============================
st.title("🛍️ Amazon Product Chatbots")

# Local LLaMA
st.subheader("🤖 Local LLaMA Chatbot (RAG + FAISS)")
user_input = st.text_input("Ask Local LLaMA:", key="local")
if user_input:
    with st.spinner("Thinking (Local LLaMA)..."):
        answer = rag_answer(user_input)
        st.markdown(f"**Assistant (Local LLaMA):** {answer}")
    with st.expander("🔍 Retrieved Context"):
        st.dataframe(retrieve_context(user_input, top_k=3))

# Groq API
st.subheader("☁️ Groq API Chatbot (Excel QA, no embeddings)")
groq_input = st.text_input("Ask Groq:", key="groq")
if groq_input:
    with st.spinner("Thinking (Groq API)..."):
        groq_answer = groq_chat_answer(groq_input, df)
        st.markdown(f"**Assistant (Groq):** {groq_answer}")
    with st.expander("📄 Excel Preview (first row only)"):
        st.dataframe(df.head(1))
