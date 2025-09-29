import streamlit as st
import pandas as pd
import torch
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq 
import os
from dotenv import load_dotenv

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


def rag_answer(user_query):
  
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
# Sidebar Navigation
# ===============================
st.sidebar.image("Images/top_banner.png", use_container_width=True)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Choose a section:", ["Chatbots", "Data Analysis"])
st.sidebar.image("Images/bottom_banner.png", use_container_width =True) 
st.title("🛍️ Amazon Product Explorer")

# ===============================
# Page 1: Chatbots
# ===============================
if page == "Chatbots":
    st.subheader("🤖 Local LLaMA Chatbot (RAG + FAISS)")
    user_input = st.text_input("Ask Local LLaMA:", key="local")
    if user_input:
        with st.spinner("Thinking (Local LLaMA)..."):
            answer = rag_answer(user_input)
            st.markdown(f"**Assistant (Local LLaMA):** {answer}")
        with st.expander("🔍 Retrieved Context"):
            st.dataframe(retrieve_context(user_input, top_k=3))

    st.subheader("☁️ Groq API Chatbot (Excel QA, no embeddings)")
    groq_input = st.text_input("Ask Groq:", key="groq")
    if groq_input:
        with st.spinner("Thinking (Groq API)..."):
            groq_answer = groq_chat_answer(groq_input, df)
            st.markdown(f"**Assistant (Groq):** {groq_answer}")
        with st.expander("📄 Excel Preview "):
            st.dataframe(df.head(5))

# ===============================
# Page 2: Data Analysis
# ===============================

elif page == "Data Analysis":
    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    df["main_category"] = df["category"].astype(str).str.split("|").str[0]


    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Products", f"{len(df):,}")
    with k2: st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    with k3: st.metric("Avg Discount (%)", f"{((1 - (df['discounted_price']/df['actual_price']))*100).mean():.1f}%")
    with k4: st.metric("Unique Categories", f"{df["main_category"].nunique():,}")

    st.markdown("---")

    #  Sentiment Distribution
    st.subheader("💡 Sentiment Distribution of Reviews")
    fig_sent = px.histogram(df, x="Feedback_review", color="Feedback_review",
                            color_discrete_sequence=["#4f008c", "#ff375e"],
                            template="plotly_white")
    st.plotly_chart(fig_sent, use_container_width=True)

    #  Ratings vs Sentiment Heatmap
    st.subheader("⭐ Ratings vs Sentiment")
    pivot = pd.crosstab(df["rating"], df["Feedback_review"]).reset_index()
    fig_heat = px.imshow(pivot.set_index("rating"),
                         text_auto=True, color_continuous_scale="YlGnBu",
                         template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Price Distribution
    st.subheader("💲 Price Distribution")
    fig_price = px.histogram(df, x="actual_price", nbins=30, 
                             color_discrete_sequence=["#4f008c"],
                             template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True)

    # Top 10 Products by Review Count
    st.subheader("🔝 Top 10 Products by Review Count")
    top_products = df["product_name"].value_counts().head(10).reset_index()
    top_products.columns = ["Product", "Reviews"]
    fig_top = px.bar(top_products.sort_values("Reviews"),
                     x="Reviews", y="Product", orientation="h",
                     text="Reviews", color="Reviews",
                     color_continuous_scale=["#4f008c", "#ff375e"],
                     template="plotly_white")
    st.plotly_chart(fig_top, use_container_width=True)

    # Average Rating per Category
    st.subheader("📦 Average Rating per Category")
    avg_ratings = df.groupby("main_category")["rating"].mean().reset_index().sort_values("rating", ascending=False).head(10)
    fig_avg = px.bar(avg_ratings.sort_values("rating"),
                     x="rating", y="main_category", orientation="h",
                     text="rating", color="rating",
                     color_continuous_scale=["#4f008c", "#ff375e"],
                     template="plotly_white")
    st.plotly_chart(fig_avg, use_container_width=True)

    #  Distribution of Features Column 
    if "features" in df.columns:
        st.subheader("⚙️ Distribution of Features")
        feature_counts = df["features"].explode().value_counts().head(10).reset_index()
        feature_counts.columns = ["Feature", "Count"]
        fig_feat = px.bar(feature_counts.sort_values("Count"),
                          x="Count", y="Feature", orientation="h",
                          text="Count", color="Count",
                          color_continuous_scale=["#4f008c", "#ff375e"],
                          template="plotly_white")
        st.plotly_chart(fig_feat, use_container_width=True)

    # Categories with Highest Discount
    st.subheader("💸 Categories with Highest Avg Discount")
    df["discount_pct"] = (1 - (df["discounted_price"] / df["actual_price"])) * 100
    discount_cat = df.groupby("main_category")["discount_pct"].mean().reset_index().sort_values("discount_pct", ascending=False).head(10)
    fig_disc = px.bar(discount_cat.sort_values("discount_pct"),
                      x="discount_pct", y="main_category", orientation="h",
                      text="discount_pct", color="discount_pct",
                      color_continuous_scale=["#4f008c", "#ff375e"],
                      template="plotly_white")
    st.plotly_chart(fig_disc, use_container_width=True)

