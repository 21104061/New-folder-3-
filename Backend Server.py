# TTS Conversion Server for Google Colab
# Run this notebook with GPU enabled (Runtime > Change runtime type > T4 GPU)

# ========================================
# STEP 0: All Python Imports
# ========================================
import os
import sys
import uuid
import tempfile
import threading
import traceback
import random
import subprocess
import json
import zipfile
import time
import io
import queue
import re
import torch
import requests

from flask import Flask, request, send_file, jsonify, Response, stream_with_context
from flask_cors import CORS

# These are imported here for the dependency check,
# but will also be used by the app.
try:
    import piper_tts
    import whisper_timestamped
    import pyngrok
except ImportError as e:
    print(f"Failed to import a critical dependency: {e}")
    # This might be run before pip install, so we don't exit yet.

# ========================================
# STEP 1: Install Dependencies
# ========================================
print("📦 Installing dependencies...")

def run_pip_install(packages):
    try:
        for pkg in packages:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", "--upgrade",
                pkg
            ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {pkg}: {e}")
        return False

print("Installing basic dependencies...")
basic_deps = [
    "flask",
    "flask-cors",
    "requests",
    "numpy",
    "torch",
    "torchaudio",
    "soundfile",
    "pyngrok"  # Added for Cloudflare tunnels
]

print("Installing TTS dependencies...")
tts_deps = [
    "https://github.com/rhasspy/piper/archive/refs/heads/master.zip",
    "https://github.com/rhasspy/piper-phonemize/archive/refs/heads/master.zip",
    "whisper-timestamped"
]

if not run_pip_install(basic_deps):
    print("⚠️ Failed to install basic dependencies")
    sys.exit(1)

if not run_pip_install(tts_deps):
    print("⚠️ Failed to install TTS dependencies")
    sys.exit(1)

# --- Check Dependencies ---
print("\nVerifying installations...")
try:
    import flask
    import flask_cors
    import whisper_timestamped
    import requests
    import torch
    import numpy
    import pyngrok
    print("✅ Basic dependencies verified")
    
    # Try importing piper (it might have a different import name)
    import piper
    print("✅ Piper TTS verified")
except ImportError as e:
    print(f"❌ Failed to verify installations: {e}")
    print("Please check the installation logs above for errors.")
    sys.exit(1)
    # Optionally, exit here if dependencies are critical
    # sys.exit("Dependency check failed.")
# ------------------------------


# ========================================
# STEP 2: Setup Tunnel
# ========================================
TUNNEL_TYPE = "cloudflare"  #@param ["cloudflare", "playit.gg"]

print(f"\n🌐 Setting up {TUNNEL_TYPE} Tunnel...")

def download_playit():
    # Define the download URL and target path
    playit_url = "https://github.com/playit-cloud/playit-agent/releases/download/v0.9.3/playit-linux-amd64"
    playit_path = "/content/playit"
    
    try:
        # Download using requests
        print("Downloading playit.gg agent...")
        response = requests.get(playit_url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(playit_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Make executable
        os.chmod(playit_path, 0o755)
        print("✅ playit.gg executable downloaded and made executable.")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download or set up playit.gg: {str(e)}")
        return False

if TUNNEL_TYPE == "playit.gg":
    if not download_playit():
        print("⚠️ Failed to download playit.gg. Please check your internet connection.")
        sys.exit(1)

# ========================================
# STEP 3: Download Piper Voice Model
# ========================================
print("\n🎤 Downloading Piper voice model (en_US-lessac-medium)...")
os.makedirs("/content/piper_models", exist_ok=True)

# Download model files using subprocess
try:
    subprocess.run([
        "wget", "-q", "-O", "/content/piper_models/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    ], check=True)
    
    subprocess.run([
        "wget", "-q", "-O", "/content/piper_models/en_US-lessac-medium.onnx.json",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    ], check=True)
    print("✅ Downloaded Piper model files successfully")
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to download Piper model files: {str(e)}")
    raise

# --- ADDED MODEL CHECK ---
PIPER_MODEL = "/content/piper_models/en_US-lessac-medium.onnx"
PIPER_CONFIG = "/content/piper_models/en_US-lessac-medium.onnx.json"
if os.path.exists(PIPER_MODEL) and os.path.exists(PIPER_CONFIG):
    print("✅ Piper voice model downloaded successfully.")
else:
    print(f"❌ Piper voice model files not found. Expected: {PIPER_MODEL}, {PIPER_CONFIG}")
    print("Please check the wget download step for errors.")
    # Optionally, exit here
    # sys.exit("Piper model download failed.")
# -------------------------


# ========================================
# STEP 4: Create Flask Server & Endpoints
# ========================================

app = Flask(__name__)
CORS(app)

WORK_DIR = "/content/conversion"
os.makedirs(WORK_DIR, exist_ok=True)

print("\n✅ Server initialized!")

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    # Also include checks for model/dependencies in health check
    model_ok = os.path.exists(PIPER_MODEL) and os.path.exists(PIPER_CONFIG)
    # Basic check for a few key libraries
    deps_ok = all(pkg in sys.modules for pkg in ['piper_tts', 'flask', 'whisper_timestamped', 'requests'])

    status_message = "TTS Conversion Server is running"
    if not model_ok:
        status_message += " - WARNING: Piper model files missing!"
    if not deps_ok:
          status_message += " - WARNING: Some dependencies missing!"


    return jsonify({
        "status": "online" if model_ok and deps_ok else "warning",
        "message": status_message,
        "gpu_available": torch.cuda.is_available(),
        "piper_model_found": model_ok,
        "dependencies_loaded": deps_ok
    })

# Main conversion endpoint
@app.route('/convert', methods=['POST'])
def convert_text_to_audio():
    # Add check for model files existence at the start of endpoints that use them
    if not os.path.exists(PIPER_MODEL) or not os.path.exists(PIPER_CONFIG):
          return jsonify({"error": "Piper voice model files are missing on the server."}), 503 # Service Unavailable

    try:
        print("\n" + "="*50)
        print("📝 NEW CONVERSION REQUEST RECEIVED")
        print("="*50)

        # Get clean text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        clean_text = data['text']
        book_title = data.get('title', 'book')

        print(f"📖 Book: {book_title}")
        print(f"📊 Text length: {len(clean_text)} characters")

        # Paths
        text_file = f"{WORK_DIR}/input.txt"
        audio_file = f"{WORK_DIR}/final_audio.wav"
        mp3_file = f"{WORK_DIR}/final_audio.mp3"
        timestamps_file = f"{WORK_DIR}/timestamps.json"
        zip_file = f"{WORK_DIR}/converted_book.zip"

        # Clean up old files
        for f in [text_file, audio_file, mp3_file, timestamps_file, zip_file]:
            if os.path.exists(f):
                os.remove(f)

        # Save text to file
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        # =====================================
        # STEP 1: Generate Audio with Piper TTS
        # =====================================
        print("\n🎵 Generating audio with Piper TTS...")
        # Removed text=True to handle potential binary output correctly in case of errors
        piper_cmd = f"piper --model {PIPER_MODEL} --output_file {audio_file} < {text_file}"
        result = subprocess.run(piper_cmd, shell=True, capture_output=True) # Removed text=True

        if result.returncode != 0:
            # Decode stderr explicitly if needed for logging, but handle potential errors
            try:
                error_message = result.stderr.decode('utf-8', errors='replace')
            except Exception:
                error_message = "Undecodable Piper stderr output"
            print(f"❌ Piper error: {error_message}")
            return jsonify({"error": "TTS generation failed", "details": error_message}), 500

        print(f"✅ Audio generated: {audio_file}")

        # Convert WAV to MP3
        print("🔄 Converting to MP3...")
        subprocess.run(f"ffmpeg -i {audio_file} -codec:a libmp3lame -qscale:a 2 {mp3_file} -y",
                       shell=True, capture_output=True)
        print(f"✅ MP3 created: {mp3_file}")

        # =====================================
        # STEP 2: Generate Timestamps with Whisper
        # =====================================
        print("\n⏱️  Generating word-level timestamps with Whisper...")

        # Load Whisper model
        try:
            # This is harder to check precisely as whisper_timestamped manages models internally.
            audio_whisper = whisper_timestamped.load_audio(mp3_file)
            model = whisper_timestamped.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

            # Transcribe with word-level timestamps
            result_whisper = whisper_timestamped.transcribe(model, audio_whisper, language="en")

            # Extract word-level timestamps
            timestamps = []
            for segment in result_whisper.get('segments', []):
                for word_info in segment.get('words', []):
                    timestamps.append({
                        "word": word_info.get('text', '').strip(),
                        "start": round(word_info.get('start', 0.0), 3),
                        "end": round(word_info.get('end', 0.0), 3)
                    })

            # Save timestamps
            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps, f, indent=2)

            print(f"✅ Timestamps generated: {len(timestamps)} words")
        except Exception as whisper_e:
             print(f"⚠️  Whisper timestamp generation failed: {whisper_e}")
             # Proceed without timestamps if Whisper fails
             timestamps = []
             if os.path.exists(timestamps_file):
                 os.remove(timestamps_file)


        # =====================================
        # STEP 3: Package Everything
        # =====================================
        print("\n📦 Packaging files...")

        # --- ADDED CHECK FOR MP3 FILE ---
        if not os.path.exists(mp3_file):
            error_msg = f"❌ Error: final_audio.mp3 was not created at {mp3_file}"
            print(error_msg)
            return jsonify({"error": "MP3 conversion failed or file missing", "details": error_msg}), 500
        # -------------------------------

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(mp3_file, 'final_audio.mp3')
            if os.path.exists(timestamps_file): # Only add timestamps if they were generated
                zipf.write(timestamps_file, 'timestamps.json')
            # Also include original text for the app
            zipf.write(text_file, 'book_text.txt')

        print(f"✅ Package created: {zip_file}")
        print("\n" + "="*50)
        print("✨ CONVERSION COMPLETE!")
        print("="*50)

        # Send the zip file
        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{book_title.replace(' ', '_')}_converted.zip"
        )

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =====================================
# Real-Time Streaming Endpoint (PREFETCHING)
# =====================================

CHUNK_DELIMITER = b"--CHUNK_BOUNDARY--"

def _produce_audio_chunk(chunk_text, index, q, model_path, config_path):
    """
    [PRODUCER] Runs in a background thread to generate one audio chunk.
    Uses subprocess.communicate() which buffers the whole chunk's audio.
    This is necessary to allow the main thread to stream the *previous* chunk.
    """
    print(f"--- 🧵 Background: Starting Generation for Chunk {index} ---")
    print(f"    ➡️  Chunk text: {chunk_text[:100]}...")

    try:
        process = subprocess.Popen(
            ["piper", "--model", model_path, "--output_file", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # communicate() writes stdin, reads stdout/stderr until EOF, and waits.
        # This blocks this thread, but not the main server thread.
        # This buffers the entire audio chunk in memory.
        audio_bytes, stderr_bytes = process.communicate(chunk_text.encode('utf-8'))
        
        return_code = process.returncode
        stderr_output = stderr_bytes.decode('utf-8', errors='replace')

        if return_code != 0:
            print(f"❌ 🧵 Background: Piper error chunk {index} (Code {return_code}): {stderr_output}")
            q.put((index, None, stderr_output, return_code))
        else:
            print(f"✅ 🧵 Background: Finished Generation for Chunk {index} ({len(audio_bytes)} bytes)")
            q.put((index, audio_bytes, stderr_output, return_code))

    except Exception as e:
        print(f"❌ 🧵 Background: EXCEPTION in producer for chunk {index}: {e}")
        traceback.print_exc()
        q.put((index, None, str(e), -1)) # Use -1 for exception return code

@app.route("/stream", methods=["POST"])
def stream():
    # Add check for model files existence at the start of endpoints that use them
    if not os.path.exists(PIPER_MODEL) or not os.path.exists(PIPER_CONFIG):
        return jsonify({"error": "Piper voice model files are missing on the server."}), 503 # Service Unavailable

    try:
        print("\n" + "=" * 50)
        print("🎧 Starting PREFETCHING real-time TTS stream")
        print(f"➡️   Client Host: {request.remote_addr}") # Log client host address
        print("=" * 50)
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        print(f"📊 Text length: {len(text)} characters")

        # === Chunk the text safely ===
        def split_text_into_chunks(text, max_chars=1200):
            # re was imported at the top
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, current = [], ""
            for s in sentences:
                if len(current) + len(s) > max_chars:
                    if current.strip(): # Only add non-empty chunks
                        chunks.append(current.strip())
                    current = s + " " # Start new chunk with current sentence
                else:
                    current += s + " "
            if current.strip(): # Add the last chunk
                chunks.append(current.strip())
            return chunks

        chunks = split_text_into_chunks(text)
        print(f"🧩 Total chunks: {len(chunks)}")
        if not chunks:
             print("⚠️ No text to process after chunking.")
             # Send a valid-but-empty stream
             chunks = [] # Ensure it's an empty list

        # === Generator that streams audio from Piper sequentially ===
        def generate():
            # Assume standard Piper WAV output: 22050 Hz, 16-bit (2 bytes per sample), mono (1 channel)
            sample_rate = 22050
            bytes_per_sample = 2
            num_channels = 1
            bytes_per_second = sample_rate * bytes_per_sample * num_channels if sample_rate > 0 and bytes_per_sample > 0 else 0 # Avoid division by zero

            cumulative_duration = 0.0 # Initialize cumulative duration

            # Send init message
            init_message_json = json.dumps({
                "type": "init",
                "session_id": str(uuid.uuid4()), # Generate a new session ID for each stream
                "total_chunks": len(chunks),
                "message": "Starting real TTS stream"
            }).encode("utf-8")
            print(f"⬆️ Yielding Init Delimiter")
            yield CHUNK_DELIMITER
            print(f"⬆️ Yielding Init Message ({len(init_message_json)} bytes)")
            yield init_message_json
            print("✅ Sent init message")

            if not chunks:
                 print("⚠️ No chunks to process. Sending end_of_stream.")
                 # Fall-through to the end_of_stream message
            else:
                generation_queue = queue.Queue()
                
                # --- PREFETCHING LOGIC ---
                # Start prefetching the FIRST chunk (index 0)
                print(f"🚀 Launching producer for chunk 1 (Index 1)")
                threading.Thread(
                    target=_produce_audio_chunk, 
                    args=(chunks[0], 1, generation_queue, PIPER_MODEL, PIPER_CONFIG), # Use 1-based index
                    daemon=True # Ensure thread dies if main app dies
                ).start()

                for i, chunk_text in enumerate(chunks, 1):
                    print(f"\n--- ➡️ CONSUMER: Waiting for Chunk {i}/{len(chunks)} from producer ---")
                    
                    # Wait for the chunk (i) to be ready
                    # queue.get() blocks until an item is available
                    result_index, audio_bytes, stderr_output, return_code = generation_queue.get()
                    
                    if result_index != i:
                        print(f"❌ CRITICAL: Queue order mismatch! Expected {i}, got {result_index}. Aborting.")
                        # Send an error and break
                        error_message_chunk = {
                           "type": "error", "chunk_index": i,
                           "message": f"Critical server error: Queue order mismatch. Aborting stream.",
                           "details": f"Expected {i} got {result_index}"
                        }
                        yield CHUNK_DELIMITER
                        yield json.dumps(error_message_chunk).encode('utf-8')
                        break
                    
                    print(f"✅ CONSUMER: Got Chunk {i}. ({'ERROR' if return_code != 0 else f'{len(audio_bytes)} bytes'})")

                    # --- START PREFETCHING THE *NEXT* CHUNK ---
                    # Do this *immediately* after getting the current one,
                    # so generation overlaps with streaming.
                    if i < len(chunks): # If this isn't the last chunk
                        next_chunk_index = i + 1
                        next_chunk_text = chunks[i] # chunks is 0-indexed, so chunks[i] is chunk i+1
                        print(f"🚀 Launching producer for chunk {next_chunk_index} (Index {next_chunk_index})")
                        threading.Thread(
                            target=_produce_audio_chunk, 
                            args=(next_chunk_text, next_chunk_index, generation_queue, PIPER_MODEL, PIPER_CONFIG),
                            daemon=True
                        ).start()
                    # -------------------------------------------

                    # --- Handle Generation Error for Current Chunk ---
                    if return_code != 0:
                        print(f"❌ Piper error during chunk {i} (Return Code {return_code}): {stderr_output}")
                        error_message_chunk = {
                            "type": "error",
                            "chunk_index": i,
                            "message": f"TTS generation failed for chunk {i}",
                            "details": stderr_output
                        }
                        yield CHUNK_DELIMITER
                        yield json.dumps(error_message_chunk).encode('utf-8')
                        yield CHUNK_DELIMITER # Add end delimiter for the error block
                        continue # Move to the next chunk

                    # --- Process and Stream the (now-buffered) audio ---
                    current_chunk_duration = len(audio_bytes) / bytes_per_second if bytes_per_second > 0 else 0
                    chunk_start_time = cumulative_duration
                    
                    timestamp_message = {
                        "type": "chunk_metadata",
                        "chunk_index": i,
                        "start_time": round(chunk_start_time, 3),
                        "duration": round(current_chunk_duration, 3), # We can calculate this now
                        "text": chunk_text,
                        "audio_size": len(audio_bytes) # We can calculate this now
                    }
                    timestamp_json_bytes = json.dumps(timestamp_message).encode('utf-8')

                    print(f"⬆️ Yielding Chunk {i} Delimiter")
                    yield CHUNK_DELIMITER
                    print(f"⬆️ Yielding Chunk {i} Metadata ({len(timestamp_json_bytes)} bytes)")
                    yield timestamp_json_bytes
                    print(f"✅ Sent metadata for chunk {i} (Actual size/duration)")

                    print(f"⬆️ Starting to yield {len(audio_bytes)} audio bytes for chunk {i}...")
                    
                    # Yield audio in sub-chunks to be nice to network buffers
                    buffer_size = 8192 # Yield in 8KB chunks
                    if len(audio_bytes) > 0:
                        for j in range(0, len(audio_bytes), buffer_size):
                            yield audio_bytes[j : j + buffer_size]
                    else:
                        # Yield nothing if audio_bytes is empty (e.g., just a space)
                        pass
                    
                    # Mark end of audio block
                    print(f"⬆️ Yielding Chunk {i} End Delimiter")
                    yield CHUNK_DELIMITER
                    print(f"✅ Finished streaming audio for chunk {i}. Total bytes: {len(audio_bytes)}. Duration: {current_chunk_duration:.3f}s")
                    
                    cumulative_duration += current_chunk_duration
                    # time.sleep(0.1) # No longer needed, as queue.get() provides the pause
            
            # --- Finishing Stream (Same as before) ---
            print("\n--- Finishing Stream ---")
            end_message = {
                "type": "end_of_stream",
                "total_duration": round(cumulative_duration, 3),
                "total_chunks": len(chunks)
            }
            end_message_json = json.dumps(end_message).encode('utf-8')
            print(f"⬆️ Yielding End of Stream Delimiter")
            yield CHUNK_DELIMITER
            print(f"⬆️ Yielding End of Stream Message ({len(end_message_json)} bytes)")
            yield end_message_json
            # Add final delimiter after the last message
            print(f"⬆️ Yielding FINAL Delimiter")
            yield CHUNK_DELIMITER
            print("🎉 All chunks processed.")

        # Set the mimetype to indicate a mixed content stream
        return Response(stream_with_context(generate()), mimetype="application/octet-stream", direct_passthrough=True)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        # Cannot return JSON here if stream has started, but try
        try:
             return jsonify({"error": str(e)}), 500
        except:
             print("❌ Could not return JSON error response, stream likely started.")
             return # Just end the connection


# ========================================
# STEP 5: Start Server and Tunnel
# ========================================

def run_flask():
    # Run Flask on port 5001
    # use_reloader=False is important when running in a separate thread
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def start_tunnel_and_run_server():
    # Kill any process running on port 5001
    print("🔪 Killing any process on port 5001...")
    # Use `command -v lsof` to check if lsof exists before using it
    lsof_exists = subprocess.run("command -v lsof", shell=True, capture_output=True).returncode == 0
    if lsof_exists:
        try:
            subprocess.run("lsof -t -i:5001 | xargs -r kill -9 || true", 
                         shell=True, check=False)
        except Exception as e:
            print(f"Warning: Failed to kill processes on port 5001: {e}")
    else:
        print("⚠️  lsof command not found, could not check for processes on port 5001.")


    # Start Flask in background thread
    print("\n🚀 Starting Flask server...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print("\n" + "="*50)
    print("🎉 FLASK SERVER IS LIVE ON PORT 5001")
    print("="*50)
    
    if TUNNEL_TYPE == "cloudflare":
        print(f"\n🌐 Now starting {TUNNEL_TYPE} tunnel...")
        try:
            public_url = pyngrok.connect(5001)
            print("\n" + "="*50)
            print("🎉 SUCCESS! Server is accessible at:")
            print(f"➡️  {public_url}")
            print("\n⚠️  Copy this URL into your Android app")
            print("="*50)
            # Keep the main thread alive to keep the tunnel open
            flask_thread.join()
        except Exception as e:
            print(f"\n❌ Error starting Cloudflare tunnel: {str(e)}")
            raise

    elif TUNNEL_TYPE == "playit.gg":
        print(f"\n🌐 Now starting {TUNNEL_TYPE} tunnel...")
        playit_process = None
        try:
            # Run playit in a way that we can capture its output
            playit_process = subprocess.Popen(
                ["/content/playit", "tunnel", "http://localhost:5001"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            print("\n" + "="*50)
            print("🔍 Waiting for playit.gg tunnel URL...")
            
            # Monitor the output to find the tunnel URL
            while True:
                output = playit_process.stdout.readline()
                if output == '' and playit_process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    # Look for the tunnel URL in the output
                    if "tunnel url:" in output.lower():
                        tunnel_url = output.split("tunnel url:")[-1].strip()
                        print("\n" + "="*50)
                        print("🎉 SUCCESS! Server is accessible at:")
                        print(f"➡️  {tunnel_url}")
                        print("\n⚠️  Copy this URL into your Android app")
                        print("="*50)
                        break
            
            # Keep the process running
            try:
                playit_process.wait()
            except KeyboardInterrupt:
                print("\nShutting down server...")
                if playit_process:
                    playit_process.terminate()
                    playit_process.wait()
                print("✅ Server shut down.")
                
        except Exception as e:
            print(f"\n❌ Error starting playit.gg tunnel: {str(e)}")
            if playit_process:
                playit_process.terminate()
                playit_process.wait()
            raise

# Start the tunnel and keep the script alive
start_tunnel_and_run_server()
