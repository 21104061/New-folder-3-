# =========================================================
# CELL 1: SETUP AND WRITE THE tts_server.py FILE
# =========================================================

# 1. Install all dependencies
print("📦 Installing dependencies (gunicorn, flask, torch, transformers)...")
!pip install -q gunicorn flask flask-cors transformers torch accelerate soundfile

# 2. Define the server code as a string
# (This is the production code from our previous step)
tts_server_code = """
import os
import torch
import soundfile as sf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForTextToSpeech
import io
import time

print("🤖 Loading DiaTTS model... (This may take a moment)")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "dia-tts/dia-tts-v2-en-us-call-center-multi-speaker"

try:
    model = AutoModelForTextToSpeech.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✅ DiaTTS Model '{model_id}' loaded successfully on {device}.")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load model: {e}")
    exit()

app = Flask(__name__)
CORS(app)
SERVER_PORT = 5002

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model_id": model_id, "device": device})

@app.route('/generate-audio', methods=['POST'])
def handle_generate_audio():
    start_time = time.time()
    try:
        data = request.json
        if not data or 'text' not in data or 'speaker_id' not in data:
            print("   -> ❌ Error: 400 - Missing 'text' or 'speaker_id'")
            return jsonify({"error": "Missing 'text' or 'speaker_id'"}), 400

        text_input = data['text']
        speaker_id = int(data.get('speaker_id', 0))

        print(f"🎤 Request: Speaker {speaker_id}, Text: \\"{text_input[:40]}...\\"")

        inputs = processor(
            text_input,
            speaker_id=speaker_id,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, do_sample=True, temperature=0.7)

        sampling_rate = output["sampling_rate"]
        waveform = output["waveform"].cpu().numpy().squeeze()

        duration = len(waveform) / sampling_rate

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sampling_rate, format='WAV')
        buffer.seek(0)

        end_time = time.time()
        print(f"   -> ✅ Success: Generated {duration:.2f}s audio in {end_time - start_time:.2f}s")

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False
        )

    except Exception as e:
        end_time = time.time()
        print(f"   -> ❌ Error: 500 - Generation failed: {e} (Req time: {end_time - start_time:.2f}s)")
        return jsonify({"error": str(e)}), 500

print("\\n✅ Production server file 'tts_server.py' is ready.")
print("   Run Gunicorn to start the server.")
"""

# 3. Write the code to the file tts_server.py
with open("tts_server.py", "w") as f:
    f.write(tts_server_code)

print("\n🎉 Setup complete. 'tts_server.py' is created.")
print("   Ready to run Cell 2.")



celll 22

# =========================================================
# CELL 2: START SERVER & TUNNEL
# =========================================================

import os
import time
import subprocess

# 1. Install cloudflared
if not os.path.exists('/usr/local/bin/cloudflared'):
    print("📦 Installing cloudflared...")
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x cloudflared-linux-amd64
    !mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
    print("✅ Cloudflared installed.")

# 2. Start the Gunicorn server IN THE BACKGROUND
# The '&' at the end is the magic: it runs the process in the background.
# We pipe the output to a log file so it doesn't flood the console.
print("🚀 Starting Gunicorn server in the background...")
server_process = subprocess.Popen(
    ['gunicorn', '-w', '4', '-b', '0.0.0.0:5002', 'tts_server:app', '--timeout', '120'],
    stdout=open('gunicorn.log', 'w'),
    stderr=subprocess.STDOUT
)
print("✅ Gunicorn is running (logs are in 'gunicorn.log')")
time.sleep(10) # Give Gunicorn time to start up

# 3. Start the Cloudflare tunnel IN THE FOREGROUND
# This command will take over the cell output and print the URL.
# This is the last command, so it's OK that it blocks.
print("🚇 Starting Cloudflare tunnel (this will take over)...")
!cloudflared tunnel --url http://localhost:5002