# Local FLAN-T5 Chatbot 🧠

A simple AI chatbot built in Python that runs **fully locally** using the `google/flan-t5-base` model from Hugging Face Transformers.  
No API keys, no billing, no external services – everything runs on your machine.

---

## ✨ Features

- 🧠 Uses FLAN-T5, an instruction-tuned language model
- 💻 Runs locally on CPU (GPU if available)
- 🌐 Simple Flask web interface (chat UI in the browser)
- 🔒 No OpenAI / Hugging Face API keys needed
- 📦 Easy to install with `pip` and `requirements.txt`

---

## 🏗 Tech Stack

- **Backend:** Python, Flask  
- **AI Model:** `google/flan-t5-base` via `transformers`  
- **Frontend:** HTML, CSS, vanilla JavaScript  
- **Environment:** Runs locally (tested on Python 3.11)

---

## 🚀 How to Run

```bash
git clone https://github.com/Harshbv/local-flan-t5-chatbot.git
cd local-flan-t5-chatbot

# install dependencies
pip install -r requirements.txt

# run the web app
python app.py
