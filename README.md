# 🧠 Semantic Cache LLM Playground – Problem Statement

📌 Objective:

Build a smart LLM assistant that can answer user questions intelligently while storing and retrieving past Q&A pairs using semantic caching. The goal is to reduce unnecessary LLM calls for semantically similar questions and optimize response time, cost, and performance.
What You Need to Build:

You are tasked with developing a semantic cache-enabled Q&A system that:

    Takes any question from a user and provides a meaningful answer using a large language model (LLM).

    Checks if a semantically similar question has already been asked before.

    If a similar question exists, retrieve the answer from cache instead of calling the LLM.

    If not, generate a new answer via the LLM and store the new question-answer pair.

    View the answer source (cache or LLM)

    See similarity score and matched question (if any)

🧰 Tools You Must Use:

    Embedding Model: OpenAI’s text-embedding-3-small via LangChain or Any Open Source from Hugging Face

    Vector Store: Pinecone with cosine similarity

    LLM: Groq’s LLaMA3 via LangChain

✅Functional Expectations:

    Embed every question and store the vector in Pinecone

    Use cosine similarity to compare new queries against existing ones

    Show if the answer was retrieved from cache or generated freshly

    Timestamp every entry for future use

🧪Sample Test Cases:

Use these to verify your cache is working correctly (ask the first, then rephrase and ask the second):

🌊 Set 1 — Oceans

Q1: How many oceans are there in the world?

Q2: What is the count of oceans on Earth?

🗼 Set 2 — Capital

Q1: What is the capital of Japan?

Q2: Which city is the capital of Japan?

🇺🇸 Set 3 — US President

Q1: Who is the current President of the USA?

Q2: Who's leading the US government right now?

❌ Cache Miss Example

Q: What are the symptoms of flu? (new topic, should call LLM)
