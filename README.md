# 🔎 IntelliWebQA

https://intelliweb.streamlit.app/

**Ask questions about any webpage — instantly and intelligently.**

---

## 🚀 What Is IntelliWebQA?

**IntelliWebQA** is a powerful research assistant that lets you input any public webpage URL, extracts its content, breaks it into digestible chunks, builds semantic embeddings, and allows you to ask natural language questions about that content — with answers and sources provided by a state-of-the-art AI.

---

## 🎯 Real-World Problem It Solves

Webpages today contain a massive amount of unstructured text. If you're a:
- 🧑‍🎓 Student doing research
- 🧑‍💼 Analyst trying to extract key insights
- 🧑‍🔬 Researcher verifying facts
- 📰 Reader wanting a quick summary

...you often don’t have time to read and digest long articles or technical pages.

**IntelliWebQA solves this by turning webpages into conversational knowledge.**

---

## 🧠 What It Does

1. Accepts 1–3 URLs from the user.
2. Loads and cleans the webpage content.
3. Splits it into manageable text chunks.
4. Generates semantic embeddings using Google Gemini’s embedding model.
5. Stores embeddings in a FAISS vector index.
6. Lets you ask **natural language questions** about the content.
7. Retrieves the most relevant chunks and uses an LLM to generate an answer **with sources.**

---

## 💡 Use Cases

- Ask about key takeaways from blog posts, whitepapers, or policy documents.
- Quickly summarize long news articles.
- Extract data from government or legal web pages.
- Investigate product descriptions or reviews.

> ⚠️ Works best on **text-heavy** webpages.

---

## 🧰 Technologies & Libraries Used

| Technology / Library             | Purpose                            |
|----------------------------------|-------------------------------------|
| `Streamlit`                      | Interactive web app UI              |
| `LangChain`                      | LLM orchestration & chaining        |
| `Google Generative AI (Gemini)`  | Embedding & language model          |
| `FAISS`                          | Vector similarity search            |
| `WebBaseLoader (LangChain)`      | Loads and parses webpage text       |
| `RecursiveCharacterTextSplitter`| Splits text into manageable chunks  |

---

## 📌 Limitations

- ⚠️ Only works with **publicly accessible** webpages.
- ❌ Pages that require login or dynamically load content via JavaScript may not be parsed correctly.
- 🧱 Extremely large pages or documents may exceed token/embedding limits.
- 📉 Currently supports **up to 3 URLs** at a time to maintain responsiveness and stability.


## 📍 Roadmap Ideas

- 📄 **Add support for PDF/Word uploads** — Expand beyond URLs to let users upload files directly.
- 🧠 **Summarization mode** — Automatically summarize full documents before Q&A.
- 💾 **Long-term memory** — Persist vector index on disk for historical retrieval.
- 💬 **Chat history & multi-turn QA** — Enable more conversational interactions with memory.
- 🔁 **Support for map-reduce QA method** — Better handling of long-form or complex documents.

