# Final Project: IITB Insti-Assist — A RAG-Powered AI Assistant for IIT Bombay

##  Introduction
Welcome to the final project of our 4 week NLP journey!
 
Every LLM you've used so far — ChatGPT, Claude, Gemini — knows a great deal about the world, but nothing at all about *your* world. Ask it about IIT Bombay's hostel allotment process, the exact grading policy for a specific course, or what WnCC's Learners' Space program covers, and it will either guess, hedge, or confidently make something up. This is the exact gap Retrieval-Augmented Generation was built to close.
 
In this project, you'll go beyond just calling an LLM API — you'll build the retrieval layer that turns a generic language model into a *specialist* for a domain of your choosing: IIT Bombay itself. By the end of the week, you'll have hands-on experience with the same architecture powering real production AI assistants — chunking, embedding, vector search, and grounded generation — plus a live demo wrapped in a simple web interface to show for it.
 
The goal is nto a perfect working system, it's a working, honest model that knows what it knows and says "I don't know" when it doesn't.

----

##  Objective
 
Build a working AI assistant that can accurately answer questions about IIT Bombay — academics, hostel life, clubs, administrative processes, or any insti-specific topic of your choice — by grounding a Large Language Model in real institute documents using Retrieval-Augmented Generation (RAG). The assistant must not hallucinate — it should answer *only* from retrieved context and clearly say "I don't know" when the answer isn't in its knowledge base.
 
---
 
##  Problem Statement
 
LLMs know nothing about IIT Bombay's specific rules, processes, or culture — they weren't trained on it. Your job is to fix that *without* fine-tuning, using RAG: retrieve relevant institute documents at query time and inject them into the prompt so the model answers with real, grounded information.
 
Participants must choose **one scope** for their assistant (to keep it focused within a week):
- **Academic Assistant** — course registration, grading policy, academic calendar, exam rules
- **Hostel & Campus Life Assistant** — hostel rules, mess menus, campus facilities
- **Council/Club Assistant** — WnCC, Gymkhana, or any specific council's activities and processes
- **General Insti Assistant** — a broader mix (harder to do well in a week, but allowed)
---
 
##  Functional Requirements
 
1. **Data ingestion** — collect at least 5 real source documents (PDFs, scraped web pages, or official text) relevant to the chosen scope
2. **Chunking + Embedding** — split documents into retrievable chunks and embed them using any embedding model
3. **Vector search** — store embeddings in a vector database/index and retrieve top-k relevant chunks per query
4. **LLM integration** — call a cloud LLM (Claude, Gemini, OpenAI, or any accessible API) with retrieved context injected into the prompt
5. **Grounded answering** — the assistant must refuse to answer (say "I don't know") if the answer isn't supported by retrieved context
6. **Source display** — show which document(s)/chunk(s) were used to generate each answer
7. **Web interface** — wrap the assistant in a simple interactive UI (Streamlit or Gradio)
---
 
##  Suggested Tech Stack (not mandatory)
 
| Component | Suggested Tool |
|---|---|
| Embeddings | `sentence-transformers` (e.g. all-MiniLM-L6-v2) |
| Vector DB | FAISS, ChromaDB, or any equivalent |
| LLM | Claude API / Gemini API / OpenAI API (or any free-tier alternative) |
| UI | Streamlit or Gradio |
| Doc parsing | `pdfplumber`, `PyPDF2`, `BeautifulSoup` for scraping |
 
---
 
##  Deliverables
 
1. **GitHub repo** with full source code and a clear `README.md` explaining setup/run instructions
3. **Write-up (2 page max)** covering:
   - Chosen scope and why
   - Data sources used
   - Chunking strategy and why
   - Known limitations / what you'd improve with more time
 
---
 
##  Bonus / Stretch Goals (optional, for extra credit)
 
- Multi-turn conversational memory (follow-up questions)
- Citation highlighting (show exact quoted chunk, not just source name)
- Support for live PDF upload by the user
- Confidence score or "grounded vs. not grounded" indicator
---
 
##  Helpful Resources
 
- [Streamlit Docs](https://docs.streamlit.io/)
- [Gradio Docs](https://www.gradio.app/docs)
