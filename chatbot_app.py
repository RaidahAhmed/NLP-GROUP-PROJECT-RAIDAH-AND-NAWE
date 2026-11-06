import re
import numpy as np
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

PDF_PATH = "Understanding-endometriosis- pack_0.pdf"          
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    " Be polite. Refuse questions unrelated to the document."
    " You are not a doctor: for personal medical concerns, tell the person to see a healthcare professional."
    " If the answer is not in the context, reply exactly: I don't know, that is not in the document.")

# The LLM runs on this computer
pipe = pipeline("text-generation", model=LLM_MODEL, device_map="auto", torch_dtype="auto")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read the PDF and keep each page's number
pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(PDF_PATH).pages)]

# Split into chunks of about 500 characters, overlapping by 100
chunks = []
for page_num, text in pages:
    text = " ".join(text.split())
    for start in range(0, len(text), 400):
        piece = text[start:start + 500]
        if len(piece) > 50:
            chunks.append({"page": page_num, "text": piece})
vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)

def retrieve(q, k=3):
    scores = vectors @ embedder.encode([q], normalize_embeddings=True)[0]
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]

def clean_history(history):
    out = []
    for m in history[-6:]:
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append({"role": m["role"], "content": content})
    return out

def chat(message, history):
    hits = retrieve(message)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + clean_history(history)
    messages.append({"role": "user", "content": (
        f"Context:\n{context}\n\n"
        f"Question: {message}\n\n"
        "Reminder: answer ONLY from the context and END with the page number in brackets OR page numbers if many for example (page 3)."
    )})
    out = pipe(messages, max_new_tokens=400, do_sample=False)
    answer = out[0]["generated_text"][-1]["content"]
    if "don't know" not in answer.lower() and not re.search(r"page\s*\d+", answer, re.IGNORECASE):
        answer += f" (page {hits[0]['page']})"
    return answer

    gr.ChatInterface(fn=chat, title="Endometriosis Info Assistant").launch()
