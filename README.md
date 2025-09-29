# 🛒 Amazon Product Chatbots

An interactive **NLP-powered Chatbot project** for analyzing Amazon product data.  
It combines **Local LLaMA (RAG + FAISS)** and **Groq API** to provide insights from Excel datasets, including **sentiment analysis, feature extraction, and product search**.

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![Chatbot](https://img.shields.io/badge/Chatbot-RAG%20%2B%20QA-purple)

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Screenshots & Explanation](#-screenshots--explanation)
   - [Groq API Chatbot](#1-groq-api-chatbot)
   - [Local LLaMA Chatbot](#2-local-llama-chatbot)
4. [Tools & Setup](#-tools--setup)
5. [Setup & Run](#-setup--run-with-conda)   
6. [Groq API Key Setup](#-groq-api-key-setup)  


---

## 🚀 Features

### 🔹 Data Cleaning & Preprocessing
- Load Excel data and clean product reviews.  
- Fix encoding issues in **price columns** (`actual_price`, `discounted_price`) by removing special symbols and converting to integers.  
- Convert **ratings** to numeric values.  
- Normalize and clean text in the `about_product` column.  

### 🔹 Feature Engineering
- Extract  features from product descriptions using **Zero-Shot Classification** (`cross-encoder/nli-distilroberta-base`).  
- Candidate labels include: *Bluetooth, Wireless, Fast Charging, High Speed, WiFi, Waterproof, Voice Assistant, Full HD, Data Sync, Flexible*.  
- Add extracted features as a new column `features`.  


### 🔹 Sentiment Analysis
- Apply sentiment analysis on customer reviews.  
- Label reviews as **positive** or **negative**.  
- Store results in a new column (`Feedback_review`).  

### 🔹 Dual Chatbot Modes
- **Local LLaMA Chatbot (RAG + FAISS):**
  - Retrieval-Augmented Generation using FAISS vector database.  

- **Groq API Chatbot (Excel QA, no embeddings):**
  - Direct QA over Excel files.  
  - Answer structured questions like *“What data is stored in the features column?”*.  

### 🔹 Interactive Dashboard
- Built with **Streamlit** for real-time interaction.  
- Dataset preview and response visualization.  

---

## 📂 Project Structure

  ```bash
 Chatbot/
│── Data/ # Excel datasets
│ ├── amazon_product_reviews.xlsx
│ └── amazon_product_reviews_updated1.xlsx
│
│── Images/ # Screenshots for README
│ ├── Screenshot-2025-09-28-190729.png
│ └── Screenshot-2025-09-28-200511.png
│
│── Notebook/ # Jupyter notebook for preprocessing
│ └── Notebook.ipynb
│
│── chatbot.py # Streamlit chatbot interface
│── environment.yml # Conda environment setup
│── requirements.txt # Pip dependencies
│── README.md # Project documentation
│── .gitignore # Ignore sensitive/unneeded files
---
 ```
## 🖼 Screenshots & Explanation

### 1. Local LLaMA Chatbot
![Local LLaMA Chatbot](Images/llama.png)

💡 **Explanation:**  
This mode uses **LLaMA locally with RAG + FAISS**.  
- The user asks: *“show me products with fast charging”*.  
- The chatbot retrieves relevant rows from the dataset where product features include *Fast Charging*.  
- It shows **price, rating, and review sentiment** for each matching product.  

---

### 2. Groq API Chatbot
![Groq Chatbot](Images/groq.png)

💡 **Explanation:**  
This mode connects to the **Groq API** for Excel QA.  
- The user asks: *“What kind of data is stored in the features column?”*  
- The chatbot responds with structured details, explaining that the **features column contains product attributes** such as *High Speed, Data Sync, Flexible, Fast Charging*.  
- An **Excel preview** is also shown to validate the response.  

---

### 3. Dataset Overview
![Dataset Overview](Images/1.png)

💡 **Explanation:**  
Shows a preview of the dataset along with **key KPIs** such as total products, average rating, average discount, and number of unique categories.

---

### 4. Sentiment Distribution
![Sentiment Distribution](Images/2.png)

💡 **Explanation:**  
Visualizes the **sentiment breakdown** of product reviews, comparing positive vs. negative reviews.

---

### 5. Average Rating per Category
![Average Rating per Category](Images/3.png)

💡 **Explanation:**  
Highlights the **top-level categories** and their **average customer ratings**, helping identify the best-performing product segments.

---

## 🛠 Tools & Setup

- **Python 3.12**
- **Streamlit** – interactive app
- **Pandas** – preprocessing & features
- **FAISS** – vector database for retrieval
- **Transformers (LLaMA)** – local LLM with RAG
- **Groq API** – external chatbot for Excel QA

---
## ⚡ Setup & Run (with Conda)

1. **Clone or Download** this project folder.

2. **Install Anaconda / Miniconda**

3. **Create a new environment and run the app**

     1. *Option A – Using Conda (recommended)*

   ```bash
   git clone  https://github.com/hayaAlkhathran/ChatbotAmazonData.git
   cd Chatbot
   conda env create -f environment.yml
   conda activate Chatbot
   streamlit run Chatbot.py
   ```

      2. *Option B – Using pip*

   ```bash
   git clone  https://github.com/hayaAlkhathran/ChatbotAmazonData.git
   cd Chatbot
   conda create -n Chatbot python=3.12 -y
   conda activate Chatbot
   pip install -r requirements.txt
   streamlit run Chatbot.py
   ```


   ## 🔑 Groq API Key Setup

This project requires a **Groq API key** to run the Groq chatbot.

1. Create a File in your project root called `.env`  
2. Add your API key in the file like this:

```env
GROQ_API_KEY = "your_api_key_here"
```
3. Add your API key in chatbot:
```chatbot
load_dotenv()  # load variables from .env

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing Groq API key in .env file")
    st.stop()
```