from flask import Flask, render_template, request, session
from PyPDF2 import PdfReader
import os
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, static_folder="static")
app.secret_key = "secret_key"
app.config["SESSION_TYPE"] = "filesystem"

API_KEY = "hf_your_actual_token_here"
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_text(text):
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def split_into_chunks(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        if page_text := page.extract_text():
            text += page_text
    return text.strip()

def is_follow_up_question(question):
    followup_keywords = ["this document", "these authors", "they", "it", "those", "same document", "this project"]
    return any(kw in question.lower() for kw in followup_keywords)

def retrieve_relevant_chunks(question, chunk_data, top_k=3):
    texts = [c['text'] for c in chunk_data]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english').fit(texts + [question])
    chunk_vectors = vectorizer.transform(texts)
    question_vector = vectorizer.transform([question])
    scores = chunk_vectors.dot(question_vector.T).toarray().flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunk_data[i] for i in top_indices]

def generate_answer(relevant_chunks, question):
    question_lower = question.lower().strip()
    if question_lower in ["hi", "hello", "hey"]:
        return "Hi there! How can I help you today?", True
    if question_lower in ["who are you", "what are you"]:
        return "I'm a PDF-based knowledge assistant here to help you understand your documents.", True
    if "thank" in question_lower:
        return "You're welcome! 😊", True
    if question_lower in ["how are you", "how are you doing"]:
        return "I'm just a bunch of code, but I'm functioning great! 😄 What can I do for you?", True
    if question_lower in ["bye", "goodbye"]:
        return "Goodbye! Have a great day! 👋", True

    last_pdf = session.get("last_pdf", "the currently viewed document")
    last_question = session.get("last_question", "")
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    prompt = f"""
You're a helpful assistant answering questions from PDF documents.

Previous question (if any): {last_question}
Current document: {last_pdf}

Use the following context to answer the question clearly and briefly.
If the answer is not in the context, reply with "Sorry, I couldn't find an answer in the documents."

Context:
{context}

Q: {question}
Answer:
"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3,
            "top_k": 5,
            "repetition_penalty": 1.1
        }
    }

    for attempt in range(3):
        try:
            res = requests.post(
            "https://router.huggingface.co/hf-inference/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            headers=headers,
            json=body
            )
            
            if res.status_code == 200:
                data = res.json()
                output = data[0]["generated_text"]
                final = output.split("Answer:")[-1].strip() if "Answer:" in output else output
                return final, False
            elif attempt == 2:
                return f"❌ Error {res.status_code}: {res.text}", False
        except Exception as e:
            if attempt == 2:
                return f"⚠ Request failed: {str(e)}", False
    return "❌ Failed after 3 attempts.", False

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    top_chunks = []
    selected_pdf = request.form.get("selected_pdf", "")
    question = request.form.get("question", "").strip()
    only_current = request.form.get("only_current_pdf") == "on"
    pdfs = os.listdir(UPLOAD_FOLDER)
    pdf_url = ""

    if request.method == "POST" and 'pdf' in request.files and request.files['pdf'].filename != '':
        file = request.files['pdf']
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        pdfs = os.listdir(UPLOAD_FOLDER)
        result = f"✅ Uploaded {file.filename}"

    if selected_pdf:
        pdf_url = f"/static/uploads/{selected_pdf}"

    if question:
        chunk_data = []
        is_followup = is_follow_up_question(question)
        explicit_pdf = ""
        for pdf in pdfs:
            base_name = os.path.splitext(pdf)[0].lower().replace(" ", "").replace("_", "").replace("-", "")
            if base_name in question.lower().replace(" ", ""):
                explicit_pdf = pdf
                break

        if only_current and selected_pdf:
            pdfs_to_use = [selected_pdf]
        elif explicit_pdf:
            pdfs_to_use = [explicit_pdf]
            selected_pdf = explicit_pdf
            session["last_pdf"] = explicit_pdf
            pdf_url = f"/static/uploads/{explicit_pdf}"
        elif is_followup:
            remembered_pdf = session.get("last_pdf")
            pdfs_to_use = [remembered_pdf] if remembered_pdf else []
        else:
            pdfs_to_use = pdfs

        if not pdfs_to_use:
            result = "❌ Please select or reference a specific PDF before asking this question."
        else:
            for pdf in pdfs_to_use:
                filepath = os.path.join(UPLOAD_FOLDER, pdf)
                text = extract_pdf_text(filepath)
                cleaned = clean_text(text)
                chunks = split_into_chunks(cleaned)
                for chunk in chunks:
                    chunk_data.append({"text": chunk, "pdf": pdf})

            if chunk_data:
                top_chunks = retrieve_relevant_chunks(question, chunk_data)
                result, is_conversational = generate_answer(top_chunks, question)
                if top_chunks and not is_conversational:
                    selected_pdf = top_chunks[0]['pdf']
                    pdf_url = f"/static/uploads/{selected_pdf}"
                    session["last_pdf"] = selected_pdf
            else:
                result = "❌ No valid content found in the selected document(s)."

    if selected_pdf:
        session["last_pdf"] = selected_pdf
    session["last_question"] = question

    return render_template(
        "index6.html",
        pdfs=pdfs,
        selected_pdf=selected_pdf,
        pdf_url=pdf_url,
        result=result,
        top_chunks=top_chunks
    )

if __name__ == "__main__":
    app.run(debug=True)