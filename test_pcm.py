# ========================================
# STEP 1: Install Dependencies
# ========================================
print("📦 Installing dependencies for Server 1...")
# We only need flask, requests (to call Server 2), and transformers (for the LLM)
!pip install -q flask flask-cors requests
!pip install -q transformers accelerate bitsandbytes

# Dependency check
try:
    import flask
    import flask_cors
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import zipfile
    import uuid
    import threading
    import traceback
    import json
    import time
    import re
    import subprocess
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("✅ Dependencies installed successfully.")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    # sys.exit("Dependency check failed.") # Commented out for Colab
    # ========================================
# STEP 2: Setup Cloudflare Tunnel (for this server)
# ========================================
import os
print("\n🌐 Setting up Cloudflare Tunnel for Server 1...")

if os.path.exists("/usr/local/bin/cloudflared"):
    print("✅ Cloudflare already installed, skipping download")
else:
    print("Downloading cloudflared...")
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    !dpkg -i cloudflared-linux-amd64.deb

    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("❌ Cloudflare Tunnel executable not found.")
        # sys.exit("Cloudflare installation failed.") # Commented out for Colab
    else:
        print("✅ Cloudflare installed successfully")
  # ========================================
# STEP 2.5: Load Local LLM (The "Brain")
# ========================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("\n🤖 Loading Qwen2-7B-Instruct model...")

model_name = "Qwen/Qwen2.5-7B-Instruct"
model = globals().get('model', None)
tokenizer = globals().get('tokenizer', None)

if model is not None and tokenizer is not None:
    print("🔁 LLM already loaded in memory; reusing existing instance.")
else:
    try:
        print("📦 Loading LLM. This may take a few minutes...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ Local LLM loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        # sys.exit("LLM failed to load. Server cannot start.") # Commented out for Colab
        # ========================================
# STEP 3: Server 2 Connection & "Casting" Logic
# ========================================
import requests
import subprocess

# ⚠️ ⚠️ ⚠️
# PASTE THE PUBLIC URL FROM YOUR SERVER 2 (CLOUD FLARE) HERE
# ⚠️ ⚠️ ⚠️
TTS_SERVER_URL = "https://soundtrack-jerry-cuisine-combination.trycloudflare.com/"

if TTS_SERVER_URL == "https://soundtrack-jerry-cuisine-combination.trycloudflare.com/":
    print("="*50)
    print("⚠️  WARNING: You must update 'TTS_SERVER_URL' with your Server 2 address.")
    print("="*50)

def call_tts_server(text, speaker_id_int, output_file):
    """
    Calls Server 2 (The "Voice") to generate audio for a single segment.
    This function is designed to be run in a thread pool.
    """
    try:
        payload = {"text": text, "speaker_id": int(speaker_id_int)}

        # Make the synchronous call. The thread pool handles the "waiting".
        response = requests.post(TTS_SERVER_URL, json=payload, timeout=90)

        if response.status_code != 200:
            print(f"   -> ❌ TTS Server Error (Speaker {speaker_id_int}): {response.text}")
            # Try a fallback with speaker 0 (narrator)
            payload = {"text": text, "speaker_id": 0}
            response = requests.post(TTS_SERVER_URL, json=payload, timeout=90)
            if response.status_code != 200:
                raise Exception(f"TTS Server failed even on fallback: {response.text}")

        # Save the received audio file
        with open(output_file, 'wb') as f:
            f.write(response.content)

        return output_file # Return the path on success

    except Exception as e:
        print(f"❌ Audio generation error calling TTS server: {e}")
        # On failure, create a short silent clip as a placeholder
        # This prevents the whole book from failing
        silence_cmd = f"ffmpeg -f lavfi -i anullsrc=r=22050:cl=mono -t 0.25 -q:a 9 -y {output_file}"
        subprocess.run(silence_cmd, shell=True, capture_output=True)
        return output_file

def assign_voice_model(gender, age_group, used_voice_ids):
    """
    Assigns a DiaTTS integer speaker ID based on character traits.
    This is the "Casting Director" logic.
    `used_voice_ids` is a set() of integers that are already taken.
    """
    # These are the 8 "voice actors" (IDs 0-7) on Server 2
    voice_pools = {
        "narrator": [0],         # Neutral voice
        "male_adult": [1, 2],    # Male voices
        "female_adult": [3, 4],  # Female voices
        "male_teen": [5],        # Male (younger)
        "female_teen": [6],      # Female (younger)
        "male_child": [5],       # Re-use teen
        "female_child": [6],     # Re-use teen
        "male_elderly": [7],     # Male (older)
        "female_elderly": [3],   # Re-use adult
        "unknown": [0]           # Default
    }

    category = f"{gender.lower()}_{age_group.lower()}"
    pool = voice_pools.get(category, voice_pools.get("unknown", voice_pools["narrator"]))

    # Try to find a unique voice from the pool
    for voice_id in pool:
        if voice_id not in used_voice_ids:
            return voice_id # Return the integer ID

    # If all unique voices are used, just rotate through the pool
    used_count = sum(1 for v in used_voice_ids if v in pool)
    return pool[used_count % len(pool)] # Return the integer ID

print("✅ Server 2 connection functions are defined.")
# ========================================
# STEP 4: LLM Analysis Logic ("Rolling Memory")
# ========================================
import threading
import json
import re

LLM_SYSTEM_PROMPT = """You are an expert text analysis engine. Your task is to convert a raw text chunk into a structured list of segments.
You must also identify any new characters.

**RULES:**
1.  **Segments:** Split the text into an ordered list of segments. A segment is EITHER "narration" OR "dialogue".
2.  **Speakers:**
    * Match dialogue speakers to the `known_cast_list`.
    * If a speaker is NOT in the list (e.g., "Alice", "Guard 1"), add them to the `new_characters` list.
    * For `new_characters`, you MUST infer `gender` ("male", "female", "unknown") and `age_group` ("child", "teen", "adult", "elderly").
3.  **Format:** Respond ONLY with a single, valid JSON object. Do not add any explanation.

**JSON OUTPUT FORMAT:**
```json
{
  "new_characters": [
    {
      "name": "CharacterName",
      "gender": "male",
      "age_group": "adult"
    }
  ],
  "segments": [
    {
      "type": "narration",
      "text": "The text of the narration."
    },
    {
      "type": "dialogue",
      "speaker_name": "CharacterName",
      "text": "The text of the dialogue."
    }
  ]
}
"""
import threading
import json
import re

class GlobalCastList:
    """
    Holds the "Rolling Memory" of all characters found so far
    and assigns them a stable voice ID.
    """
    def __init__(self):
        self.characters = {} # "Alice" -> {"gender": "f", "age": "t", "voice_id": 6}
        self.used_voice_ids = set()
        self.lock = threading.Lock()

        # Add the narrator by default
        self.get_or_assign_voice("narrator", "neutral", "adult")

    def get_known_cast_prompt(self):
        """Generates the prompt string of known characters."""
        if len(self.characters) <= 1: # Only narrator
            return "No known characters yet."

        prompt = "Known Cast List:\n"
        for name, data in self.characters.items():
            if name == "narrator": continue
            prompt += f"- {name} ({data['gender']}, {data['age_group']})\n"
        return prompt

    def get_or_assign_voice(self, name, gender, age_group):
        """
        The core "casting" logic.
        If we know the character, return their ID.
        If they are new, assign them a voice ID and save it.
        """
        with self.lock:
            if name in self.characters:
                return self.characters[name]['voice_id']

            # New character! Let's cast them.
            voice_id = assign_voice_model(gender, age_group, self.used_voice_ids)

            self.characters[name] = {
                "gender": gender,
                "age_group": age_group,
                "voice_id": voice_id
            }
            self.used_voice_ids.add(voice_id)

            print(f"   [Casting] New character '{name}' ({gender}, {age_group}) assigned to Voice ID {voice_id}")
            return voice_id

def smart_chunk_text(text, max_chunk_size=3000, overlap=300):
    """
    Intelligently chunk text at paragraph boundaries.
    """
    chunks = []
    current_pos = 0

    while current_pos < len(text):
        chunk_end = min(current_pos + max_chunk_size, len(text))

        if chunk_end < len(text):
            # Try to find a paragraph break
            para_break = text.rfind('\n\n', current_pos, chunk_end)
            if para_break > current_pos + (max_chunk_size // 2):
                chunk_end = para_break
            else:
                # Try to find a sentence break
                sent_break = max(
                    text.rfind('. ', current_pos, chunk_end),
                    text.rfind('! ', current_pos, chunk_end),
                    text.rfind('? ', current_pos, chunk_end)
                )
                if sent_break > current_pos + (max_chunk_size // 2):
                    chunk_end = sent_break + 1

        chunk_text = text[current_pos:chunk_end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        current_pos = chunk_end - overlap if chunk_end < len(text) else len(text)

    return chunks

def _analyze_chunk_llm(chunk_text, known_cast_prompt):
    """Helper function to call the LLM for one chunk."""
    try:
        user_prompt = f"{known_cast_prompt}\n\nAnalyze this text chunk:\n\n{chunk_text}"
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Clean and parse the JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError(f"No JSON object found in LLM response: {response}")

        json_str = response[json_start:json_end]
        json_str = json_str.strip().strip("```json").strip("```")

        result = json.loads(json_str)
        return result

    except Exception as e:
        print(f"   ❌ LLM chunk analysis failed: {e}")
        return None

def run_llm_analysis_pass(full_text):
    """
    Implements the "Rolling Memory" method.
    Pass 1: Analyze text, build cast list, and create segment list.
    """
    print("🧠 Starting LLM Analysis Pass (Rolling Memory)...")
    global_cast = GlobalCastList()
    all_segments = []

    chunks = smart_chunk_text(full_text)

    for i, chunk in enumerate(chunks):
        print(f"   Analyzing chunk {i+1}/{len(chunks)}...")

        # 1. Get the current "memory"
        known_cast_prompt = global_cast.get_known_cast_prompt()

        # 2. Call LLM for this chunk
        result = _analyze_chunk_llm(chunk, known_cast_prompt)

        if not result or "segments" not in result:
            print(f"   ⚠️ Skipping chunk {i+1} due to analysis error.")
            continue

        # 3. Update "memory" with new characters
        for char in result.get("new_characters", []):
            global_cast.get_or_assign_voice(char['name'], char['gender'], char['age_group'])

        # 4. Add segments to the master list
        for segment in result["segments"]:
            if segment['type'] == 'narration':
                # Assign narration to the default narrator
                segment['speaker_name'] = "narrator"

            # Ensure the speaker exists in our cast, even if the LLM forgot to add them
            if segment['speaker_name'] not in global_cast.characters:
                print(f"   ⚠️ LLM used speaker '{segment['speaker_name']}' without defining. Adding as 'unknown'.")
                global_cast.get_or_assign_voice(segment['speaker_name'], "unknown", "adult")

            all_segments.append(segment)

    print(f"✅ LLM Analysis complete. Found {len(all_segments)} segments and {len(global_cast.characters)} characters.")
    return all_segments, global_cast.characters # (segments, speaker_metadata)

print("✅ LLM analysis and 'Rolling Memory' functions are defined.")
# ========================================
# STEP 5: Main Server Logic
# ========================================
import os
import subprocess
import zipfile
import uuid
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

WORK_DIR = "/content/conversion_jobs"
os.makedirs(WORK_DIR, exist_ok=True)

# --- Audio Concatenation Helper ---
def concatenate_audio_segments(segment_files, output_file):
    """Concatenate audio segments using ffmpeg."""
    try:
        concat_list_file = f"{output_file}.concat.txt"
        with open(concat_list_file, 'w', encoding='utf-8') as f:
            for seg_file in segment_files:
                if not os.path.exists(seg_file):
                    print(f"   ⚠️ Missing segment file, skipping: {seg_file}")
                    continue
                f.write(f"file '{os.path.abspath(seg_file)}'\n")

        # We need to convert the final output from WAV to MP3
        # Let's do it in one step
        mp3_file = output_file.replace(".wav", ".mp3")
        ffmpeg_cmd_mp3 = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-codec:a", "libmp3lame", "-qscale:a", "2", "-y", mp3_file
        ]

        result = subprocess.run(ffmpeg_cmd_mp3, capture_output=True, text=True)

        if os.path.exists(concat_list_file):
            os.remove(concat_list_file)

        if result.returncode != 0:
            print(f"❌ Concatenation/MP3 failed: {result.stderr}")
            return False, None

        return True, mp3_file

    except Exception as e:
        print(f"❌ Concatenation error: {e}")
        return False, None

# --- Job Management ---
active_jobs = {}
completed_jobs = {}
job_lock = threading.Lock()
conversion_executor = ThreadPoolExecutor(max_workers=2) # Max 2 books at a time

def process_conversion_async(job_id, clean_text, book_title):
    """
    This is the main background task for a book.
    """
    try:
        print(f"\n🚀 Starting async conversion for job {job_id}")
        job_dir = f"{WORK_DIR}/job_{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        text_file = f"{job_dir}/input.txt"
        speakers_metadata_file = f"{job_dir}/speakers_metadata.json"
        zip_file = f"{job_dir}/converted_book.zip"

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        # ===================================
        # PASS 1: LLM Analysis (Rolling Memory)
        # ===================================
        with job_lock:
            active_jobs[job_id]['progress'] = 'Analyzing text with LLM...'

        segments, speaker_metadata = run_llm_analysis_pass(clean_text)

        if not segments:
            raise Exception("LLM analysis failed, no segments produced.")

        # Save speaker metadata
        speakers_metadata_json = {
            "mode": "multi_voice",
            "llm_powered": True,
            "llm_model": model_name,
            "speakers": speaker_metadata,
            "segments_count": len(segments)
        }
        with open(speakers_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(speakers_metadata_json, f, indent=2)

        # ===================================
        # PASS 2: Parallel Audio Generation (Calling Server 2)
        # ===================================
        print(f"\n🎵 Generating {len(segments)} audio segments in parallel (10 workers)...")
        with job_lock:
            active_jobs[job_id]['progress'] = 'Generating audio segments...'

        segment_audio_files = [None] * len(segments) # Pre-allocate list

        # Use a ThreadPoolExecutor to send multiple requests to Server 2 at once
        # We use max_workers=10 to hit Server 2 hard
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {}

            for i, segment in enumerate(segments):
                segment_text = segment['text']
                if not segment_text.strip():
                    continue

                speaker_name = segment['speaker_name']
                # Look up the integer voice ID from our "casting sheet"
                voice_id_int = speaker_metadata[speaker_name]['voice_id']
                segment_audio_file = f"{job_dir}/segment_{i:04d}.wav"

                future = executor.submit(
                    call_tts_server,
                    segment_text,
                    voice_id_int,
                    segment_audio_file
                )
                future_to_index[future] = i

            # Collect results as they complete
            for i_completed, future in enumerate(as_completed(future_to_index)):
                index = future_to_index[future]
                try:
                    result_path = future.result()
                    segment_audio_files[index] = result_path

                    if (i_completed + 1) % 20 == 0: # Update every 20 segments
                        progress_pct = int(((i_completed + 1) / len(segments)) * 100)
                        print(f"   ...Audio Progress: {i_completed+1}/{len(segments)} ({progress_pct}%)")
                        with job_lock:
                            active_jobs[job_id]['progress'] = f'Generating audio... ({progress_pct}%)'
                except Exception as e:
                    print(f"   ❌ Segment {index} failed: {e}")

        # Filter out any None entries (silent/empty segments)
        segment_files_ordered = [f for f in segment_audio_files if f is not None and os.path.exists(f)]

        print("\n🔗 Concatenating all segments into MP3...")
        with job_lock:
            active_jobs[job_id]['progress'] = 'Combining audio segments...'

        final_audio_wav = f"{job_dir}/final_audio.wav" # Temp name
        success, final_mp3_file = concatenate_audio_segments(segment_files_ordered, final_audio_wav)

        if not success:
            raise Exception("Audio concatenation failed")

        # Cleanup segment files
        for seg_file in segment_files_ordered:
            if os.path.exists(seg_file):
                os.remove(seg_file)

        # ===================================
        # PASS 3: Package Files
        # ===================================
        with job_lock:
            active_jobs[job_id]['progress'] = 'Packaging files...'

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(final_mp3_file, 'final_audio.mp3')
            zipf.write(text_file, 'book_text.txt')
            zipf.write(speakers_metadata_file, 'speakers_metadata.json')

        # Mark job as completed
        with job_lock:
            job_data = active_jobs.pop(job_id)
            completed_jobs[job_id] = {
                'status': 'completed',
                'zip_file': zip_file,
                'title': book_title,
                'completed_at': time.time(),
                'started_at': job_data['started_at']
            }
        print(f"✅ Async conversion completed for job {job_id}")

    except Exception as e:
        print(f"❌ Async conversion FAILED for job {job_id}: {e}")
        traceback.print_exc()
        with job_lock:
            job_data = active_jobs.pop(job_id, {})
            completed_jobs[job_id] = {
                'status': 'failed',
                'error': str(e),
                'title': book_title,
                'completed_at': time.time(),
                'started_at': job_data.get('started_at', time.time())
            }

print("✅ Main server logic and job processor functions are defined.")
# ========================================
# STEP 5: Main Server Logic
# ========================================
import os
import subprocess
import zipfile
import uuid
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

WORK_DIR = "/content/conversion_jobs"
os.makedirs(WORK_DIR, exist_ok=True)

# --- Audio Concatenation Helper ---
def concatenate_audio_segments(segment_files, output_file):
    """Concatenate audio segments using ffmpeg."""
    try:
        concat_list_file = f"{output_file}.concat.txt"
        with open(concat_list_file, 'w', encoding='utf-8') as f:
            for seg_file in segment_files:
                if not os.path.exists(seg_file):
                    print(f"   ⚠️ Missing segment file, skipping: {seg_file}")
                    continue
                f.write(f"file '{os.path.abspath(seg_file)}'\n")

        # We need to convert the final output from WAV to MP3
        # Let's do it in one step
        mp3_file = output_file.replace(".wav", ".mp3")
        ffmpeg_cmd_mp3 = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-codec:a", "libmp3lame", "-qscale:a", "2", "-y", mp3_file
        ]

        result = subprocess.run(ffmpeg_cmd_mp3, capture_output=True, text=True)

        if os.path.exists(concat_list_file):
            os.remove(concat_list_file)

        if result.returncode != 0:
            print(f"❌ Concatenation/MP3 failed: {result.stderr}")
            return False, None

        return True, mp3_file

    except Exception as e:
        print(f"❌ Concatenation error: {e}")
        return False, None

# --- Job Management ---
active_jobs = {}
completed_jobs = {}
job_lock = threading.Lock()
conversion_executor = ThreadPoolExecutor(max_workers=2) # Max 2 books at a time

def process_conversion_async(job_id, clean_text, book_title):
    """
    This is the main background task for a book.
    """
    try:
        print(f"\n🚀 Starting async conversion for job {job_id}")
        job_dir = f"{WORK_DIR}/job_{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        text_file = f"{job_dir}/input.txt"
        speakers_metadata_file = f"{job_dir}/speakers_metadata.json"
        zip_file = f"{job_dir}/converted_book.zip"

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        # ===================================
        # PASS 1: LLM Analysis (Rolling Memory)
        # ===================================
        with job_lock:
            active_jobs[job_id]['progress'] = 'Analyzing text with LLM...'

        segments, speaker_metadata = run_llm_analysis_pass(clean_text)

        if not segments:
            raise Exception("LLM analysis failed, no segments produced.")

        # Save speaker metadata
        speakers_metadata_json = {
            "mode": "multi_voice",
            "llm_powered": True,
            "llm_model": model_name,
            "speakers": speaker_metadata,
            "segments_count": len(segments)
        }
        with open(speakers_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(speakers_metadata_json, f, indent=2)

        # ===================================
        # PASS 2: Parallel Audio Generation (Calling Server 2)
        # ===================================
        print(f"\n🎵 Generating {len(segments)} audio segments in parallel (10 workers)...")
        with job_lock:
            active_jobs[job_id]['progress'] = 'Generating audio segments...'

        segment_audio_files = [None] * len(segments) # Pre-allocate list

        # Use a ThreadPoolExecutor to send multiple requests to Server 2 at once
        # We use max_workers=10 to hit Server 2 hard
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {}

            for i, segment in enumerate(segments):
                segment_text = segment['text']
                if not segment_text.strip():
                    continue

                speaker_name = segment['speaker_name']
                # Look up the integer voice ID from our "casting sheet"
                voice_id_int = speaker_metadata[speaker_name]['voice_id']
                segment_audio_file = f"{job_dir}/segment_{i:04d}.wav"

                future = executor.submit(
                    call_tts_server,
                    segment_text,
                    voice_id_int,
                    segment_audio_file
                )
                future_to_index[future] = i

            # Collect results as they complete
            for i_completed, future in enumerate(as_completed(future_to_index)):
                index = future_to_index[future]
                try:
                    result_path = future.result()
                    segment_audio_files[index] = result_path

                    if (i_completed + 1) % 20 == 0: # Update every 20 segments
                        progress_pct = int(((i_completed + 1) / len(segments)) * 100)
                        print(f"   ...Audio Progress: {i_completed+1}/{len(segments)} ({progress_pct}%)")
                        with job_lock:
                            active_jobs[job_id]['progress'] = f'Generating audio... ({progress_pct}%)'
                except Exception as e:
                    print(f"   ❌ Segment {index} failed: {e}")

        # Filter out any None entries (silent/empty segments)
        segment_files_ordered = [f for f in segment_audio_files if f is not None and os.path.exists(f)]

        print("\n🔗 Concatenating all segments into MP3...")
        with job_lock:
            active_jobs[job_id]['progress'] = 'Combining audio segments...'

        final_audio_wav = f"{job_dir}/final_audio.wav" # Temp name
        success, final_mp3_file = concatenate_audio_segments(segment_files_ordered, final_audio_wav)

        if not success:
            raise Exception("Audio concatenation failed")

        # Cleanup segment files
        for seg_file in segment_files_ordered:
            if os.path.exists(seg_file):
                os.remove(seg_file)

        # ===================================
        # PASS 3: Package Files
        # ===================================
        with job_lock:
            active_jobs[job_id]['progress'] = 'Packaging files...'

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(final_mp3_file, 'final_audio.mp3')
            zipf.write(text_file, 'book_text.txt')
            zipf.write(speakers_metadata_file, 'speakers_metadata.json')

        # Mark job as completed
        with job_lock:
            job_data = active_jobs.pop(job_id)
            completed_jobs[job_id] = {
                'status': 'completed',
                'zip_file': zip_file,
                'title': book_title,
                'completed_at': time.time(),
                'started_at': job_data['started_at']
            }
        print(f"✅ Async conversion completed for job {job_id}")

    except Exception as e:
        print(f"❌ Async conversion FAILED for job {job_id}: {e}")
        traceback.print_exc()
        with job_lock:
            job_data = active_jobs.pop(job_id, {})
            completed_jobs[job_id] = {
                'status': 'failed',
                'error': str(e),
                'title': book_title,
                'completed_at': time.time(),
                'started_at': job_data.get('started_at', time.time())
            }

print("✅ Main server logic and job processor functions are defined.")
# ========================================
# STEP 6: Flask Endpoints (Async only)
# ========================================
from flask import send_file

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Server 1 (LLM Brain) is running.",
        "llm_model": model_name
    })

@app.route('/convert-async', methods=['POST'])
def convert_text_to_audio_async():
    """Starts the async conversion job."""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        clean_text = data['text']
        book_title = data.get('title', 'book')
        job_id = str(uuid.uuid4())

        with job_lock:
            active_jobs[job_id] = {
                'status': 'queued',
                'progress': 'Queued for processing...',
                'title': book_title,
                'started_at': time.time()
            }

        # Start the background task
        conversion_executor.submit(process_conversion_async, job_id, clean_text, book_title)

        print(f"✅ Job {job_id} queued for async processing")
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Conversion started. Use /status/<job_id> to check progress."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Gets status of async conversion job"""
    with job_lock:
        if job_id in active_jobs:
            job_data = active_jobs[job_id]
            return jsonify({
                "job_id": job_id,
                "status": job_data['status'],
                "progress": job_data['progress'],
                "elapsed_time": time.time() - job_data['started_at']
            })
        if job_id in completed_jobs:
            job_data = completed_jobs[job_id]
            response = {
                "job_id": job_id,
                "status": job_data['status'],
                "total_time": job_data['completed_at'] - job_data['started_at']
            }
            if job_data['status'] == 'failed':
                response['error'] = job_data['error']
            else:
                response['download_url'] = f"/download/{job_id}"
            return jsonify(response)
    return jsonify({"error": "Job not found"}), 404

@app.route('/download/<job_id>', methods=['GET'])
def download_converted_file(job_id):
    """Download completed conversion file"""
    with job_lock:
        if job_id not in completed_jobs or completed_jobs[job_id]['status'] != 'completed':
            return jsonify({"error": "Job not found or not completed"}), 404

        zip_file = completed_jobs[job_id]['zip_file']
        title = completed_jobs[job_id]['title']

        if not os.path.exists(zip_file):
            return jsonify({"error": "Converted file not found"}), 404

        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{title.replace(' ', '_')}_audiobook.zip"
        )

print("✅ Flask API endpoints are defined.")
# ========================================
# STEP 7: Start Server and Tunnel
# ========================================
import threading
import time

def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

# Start flask
print("\n🚀 Starting Flask server for Server 1...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(3) # Give flask time to start

# Start cloudflare tunnel
print("\n🚇 Starting Cloudflare tunnel for Server 1...")
print("   This cell will now block and run forever.")
print("   Your public URL for Server 1 will appear below:")
# The ! starts this in the foreground and will block the cell,
# which is what we want to keep the Colab instance alive.
!cloudflared tunnel --url http://localhost:5001