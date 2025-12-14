from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# ---------- MODEL LOADING ----------
MODEL_NAME = "google/flan-t5-base"  # you already downloaded this

print("🔄 Loading model (first time can be slow)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def chat_with_bot(user_input: str) -> str:
    prompt = (
        "You are a polite, accurate teacher explaining things to a beginner.\n"
        "Give a clear explanation in 2–4 simple sentences. Avoid repeating the same sentence.\n\n"
        f"Question: {user_input}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # remove exact duplicate sentences
    sentences = [s.strip() for s in reply.split('.') if s.strip()]
    unique = []
    for s in sentences:
        if s not in unique:
            unique.append(s)
    cleaned = '. '.join(unique)
    if cleaned and not cleaned.endswith('.'):
        cleaned += '.'

    return cleaned or reply


# ---------- FLASK ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please type something 😄"}), 400

    reply = chat_with_bot(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    # debug=True for development only
    app.run(debug=True)
