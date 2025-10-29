# TTS Conversion Server for Google Colab with Local LLM (gpt-oss-20b)
# Run this notebook with GPU enabled (Runtime > Change runtime type > T4 GPU or A100)

import os
import sys
import uuid
import threading
import traceback
import json
import time
import re
from flask import stream_with_context
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================================
# STEP 1: Install Dependencies
# ========================================
print("📦 Installing dependencies...")
!pip install -q piper-tts flask flask-cors whisper-timestamped requests
!pip install -q transformers accelerate bitsandbytes

# Dependency check
try:
    import flask
    import flask_cors
    import whisper_timestamped
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Check if piper command is available (it's a CLI tool, not a Python module)
    piper_check = subprocess.run("which piper", shell=True, capture_output=True)
    if piper_check.returncode != 0:
        print("⚠️ Piper TTS not found in PATH, will be installed with piper-tts package")

    print("✅ Dependencies installed successfully.")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit("Dependency check failed.")

# ========================================
# STEP 2: Setup Cloudflare Tunnel
# ========================================
print("\n🌐 Setting up Cloudflare Tunnel...")

if os.path.exists("/usr/local/bin/cloudflared"):
    print("✅ Cloudflare already installed, skipping download")
else:
    print("Downloading cloudflared...")
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    !dpkg -i cloudflared-linux-amd64.deb

    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("❌ Cloudflare Tunnel executable not found.")
        sys.exit("Cloudflare installation failed.")
    print("✅ Cloudflare installed successfully")

# ========================================
# STEP 2.5: Load Local LLM
# ========================================
print("\n🤖 Loading Qwenw2.5-7B model...")
print("   Model size: ~13.8GB")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Reuse previously loaded model/tokenizer if available (prevents re-loading in same process)
model = globals().get('model', None)
tokenizer = globals().get('tokenizer', None)

if model is not None and tokenizer is not None:
    print("🔁 LLM already loaded in memory; reusing existing instance.")
else:
    try:
        # Use Transformers' built-in caching; if files exist, they won't re-download
        print("📦 Using HuggingFace cache if available...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Prefer 4-bit quantization with auto device map; fallback to 8-bit with CPU offload; then CPU
        model = None
        if torch.cuda.is_available():
            try:
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
            except Exception as e1:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                except Exception as e2:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                    )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        print("✅ Local LLM loaded successfully!")
        print(f"   Model: {model_name}")
        if torch.cuda.is_available():
            try:
                print(f"   CUDA: {torch.cuda.get_device_name(0)} • {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            except Exception:
                pass
        else:
            print("   CUDA: not available; using CPU/offload")

    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        # Graceful degrade: continue without LLM (single-voice mode)
        model = None
        tokenizer = None
        print("⚠️ LLM disabled. Server will run in single-voice mode without analysis.")

# ========================================
# STEP 3: Download Piper Voice Models
# ========================================
print("\n🎤 Setting up Piper voice models...")
os.makedirs("/content/piper_models", exist_ok=True)

# Base narrator voice
base_model = "/content/piper_models/en_US-lessac-medium.onnx"
base_config = "/content/piper_models/en_US-lessac-medium.onnx.json"

if os.path.exists(base_model) and os.path.exists(base_config):
    print("✅ Base voice model already exists, skipping download")
else:
    print("Downloading base voice model...")
    !wget -q -O /content/piper_models/en_US-lessac-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    !wget -q -O /content/piper_models/en_US-lessac-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    print("✅ Base voice model downloaded")

print("\n🎭 Downloading additional voice models...")

voice_models = [
    ("en_US-danny-medium", "en/en_US/danny/medium/"),
    ("en_US-hfc_male-medium", "en/en_US/hfc_male/medium/"),
    ("en_GB-alan-medium", "en/en_GB/alan/medium/"),
    ("en_US-amy-medium", "en/en_US/amy/medium/"),
    ("en_GB-jenny_dioco-medium", "en/en_GB/jenny_dioco/medium/"),
    ("en_GB-alba-medium", "en/en_GB/alba/medium/"),
    ("en_US-bryce-medium", "en/en_US/bryce/medium/"),
    ("en_GB-northern_english_male-medium", "en/en_GB/northern_english_male/medium/"),
    ("en_US-hfc_female-medium", "en/en_US/hfc_female/medium/"),
    ("en_GB-cori-medium", "en/en_GB/cori/medium/"),
    ("en_US-arctic-medium", "en/en_US/arctic/medium/"),
    ("en_GB-aru-medium", "en/en_GB/aru/medium/")
]

base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
downloaded_count = 0

for model_name_voice, path in voice_models:
    model_file = f"/content/piper_models/{model_name_voice}.onnx"
    config_file = f"/content/piper_models/{model_name_voice}.onnx.json"

    if os.path.exists(model_file) and os.path.exists(config_file):
        downloaded_count += 1
        continue

    try:
        model_url = f"{base_url}{path}{model_name_voice}.onnx"
        config_url = f"{base_url}{path}{model_name_voice}.onnx.json"

        os.system(f"wget -q -O {model_file} {model_url}")
        os.system(f"wget -q -O {config_file} {config_url}")

        if os.path.exists(model_file) and os.path.exists(config_file):
            downloaded_count += 1
    except Exception as e:
        print(f"   ❌ Error downloading {model_name_voice}: {e}")

print(f"\n✅ Voice models ready! Downloaded {downloaded_count} new models")

PIPER_MODEL = "/content/piper_models/en_US-lessac-medium.onnx"
PIPER_CONFIG = "/content/piper_models/en_US-lessac-medium.onnx.json"

# ========================================
# STEP 4: Create Flask Server with Parallel Pipeline
# ========================================
from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
import zipfile
import whisper_timestamped as whisper
import io

app = Flask(__name__)
CORS(app)

WORK_DIR = "/content/conversion"
os.makedirs(WORK_DIR, exist_ok=True)

print("\n✅ Server initialized!")

# ========================================
# Advanced Parallel Pipeline System
# ========================================

# System Prompt for Local LLM
SYSTEM_PROMPT = """You are a text analysis module for voice synthesis. Analyze the story text and extract speaker information.

**Task:**
1. Identify all unique speakers who have spoken lines
2. For each speaker determine:
   - Name (exact if known, or Speaker_1, Speaker_2, etc.)
   - Gender: "male", "female", or "unknown"
   - Age_group: "child", "teen", "adult", "elderly", or "unknown"
3. Extract each spoken dialogue line with its speaker

**IMPORTANT:** You're analyzing a CHUNK of a larger story. Speakers may have been introduced in previous chunks.
- If you see pronouns without names, use descriptive IDs like "Speaker_Male_1", "Speaker_Female_1"
- Be consistent with speaker identification across chunks
- Include context clues to help identify speakers across chunks

**Rules:**
- Use "unknown" if gender/age cannot be inferred
- Use descriptive IDs for unnamed speakers (e.g., "Speaker_Male_1", "Old_Woman_1")
- Include ONLY spoken dialogue, not narration
- Look for contextual clues (pronouns, descriptions, names)

**Output ONLY valid JSON in this exact format:**
```json
{
  "characters": [
    {
      "id": "Speaker_1",
      "name": "Alice",
      "gender": "female",
      "age_group": "teen",
      "description": "Young girl, protagonist"
    }
  ],
  "dialogues": [
    {
      "speaker_id": "Speaker_1",
      "line": "Hello there!",
      "context_before": "Alice smiled and said,",
      "context_after": "She waved goodbye."
    }
  ]
}
```

Return ONLY the JSON, no explanations."""

class CharacterRegistry:
    """Global character registry to maintain consistency across chunks."""
    def __init__(self):
        self.characters = {}  # id -> character_info
        self.lock = threading.Lock()

    def merge_character(self, char_info):
        """Merge or add character to registry."""
        with self.lock:
            char_id = char_info['id']

            # Try to find existing character by name or description
            existing_id = None
            for cid, existing_char in self.characters.items():
                if existing_char.get('name') == char_info.get('name') and char_info.get('name') != 'Unknown':
                    existing_id = cid
                    break

            if existing_id:
                # Update existing character with new info
                self.characters[existing_id].update({
                    k: v for k, v in char_info.items()
                    if v not in ['unknown', 'Unknown', None]
                })
                return existing_id
            else:
                # Add new character
                self.characters[char_id] = char_info
                return char_id

    def get_all_characters(self):
        """Get all registered characters."""
        with self.lock:
            return list(self.characters.values())

def smart_chunk_text(text, max_chunk_size=6000, overlap=300):
    """
    Intelligently chunk text at paragraph/sentence boundaries with overlap.
    Overlap helps maintain speaker context across chunks.
    """
    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Calculate chunk end
        chunk_end = min(current_pos + max_chunk_size, len(text))

        # If not at end, try to break at paragraph
        if chunk_end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', current_pos, chunk_end)
            if para_break > current_pos + max_chunk_size // 2:
                chunk_end = para_break
            else:
                # Look for sentence break
                sent_break = max(
                    text.rfind('. ', current_pos, chunk_end),
                    text.rfind('! ', current_pos, chunk_end),
                    text.rfind('? ', current_pos, chunk_end)
                )
                if sent_break > current_pos + max_chunk_size // 2:
                    chunk_end = sent_break + 1

        # Extract chunk
        chunk = text[current_pos:chunk_end].strip()
        if chunk:
            chunks.append({
                'text': chunk,
                'start_pos': current_pos,
                'end_pos': chunk_end
            })

        # Move to next chunk with overlap
        current_pos = chunk_end - overlap if chunk_end < len(text) else len(text)

    return chunks

def analyze_chunk_with_llm(chunk_text, chunk_index, total_chunks, registry):
    """
    Analyze a single chunk with LLM.
    Returns: (characters, dialogues, chunk_index)
    """
    # If LLM isn't available, skip and return no analysis
    if 'model' not in globals() or model is None or 'tokenizer' not in globals() or tokenizer is None:
        return ([], [], chunk_index)

    print(f"   🔍 Analyzing chunk {chunk_index + 1}/{total_chunks} ({len(chunk_text)} chars)...")

    try:
        # Prepare context about known characters
        known_chars_context = ""
        if registry.characters:
            known_chars_context = "\n\nKnown characters from previous chunks:\n"
            for char in list(registry.characters.values())[:10]:  # Limit to avoid token overflow
                known_chars_context += f"- {char.get('name', char['id'])}: {char.get('gender', 'unknown')} {char.get('age_group', 'unknown')}\n"

        # Prepare prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text chunk ({chunk_index + 1}/{total_chunks}):{known_chars_context}\n\n{chunk_text}"}
        ]

        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # **IMPROVEMENT**: Find and clean JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in LLM response")

        json_str = response[json_start:json_end]

        # Clean common LLM errors
        json_str = json_str.strip().strip("```json").strip("```") # Remove markdown
        json_str = re.sub(r"\\'", "'", json_str) # Fix escaped single quotes
        json_str = re.sub(r"\n", " ", json_str) # Remove newlines that break strings
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str) # Fix trailing commas

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"       ⚠️ JSON decode failed even after cleaning: {e}")
            print(f"       FAILED_JSON: {json_str[:500]}") # Log first 500 chars
            raise ValueError(f"No valid JSON after attempting fixes: {e}")

        if "characters" not in result or "dialogues" not in result:
            raise ValueError("Invalid JSON structure")

        # Merge characters into registry
        for char in result["characters"]:
            char_id = registry.merge_character(char)
            # Update dialogue speaker_ids to use registry IDs
            for dialogue in result["dialogues"]:
                if dialogue["speaker_id"] == char["id"]:
                    dialogue["speaker_id"] = char_id

        print(f"       Succeeded ✅ Found {len(result['characters'])} chars, {len(result['dialogues'])} dialogues")

        return (result["characters"], result["dialogues"], chunk_index)

    except Exception as e:
        print(f"       Failed ❌ Chunk {chunk_index + 1} analysis failed: {e}")
        traceback.print_exc() # Print full error for debugging
        return ([], [], chunk_index)

def assign_voice_model(gender, age_group, speaker_id, used_voices):
    """Assign voice model based on gender and age."""
    voice_pools = {
        "narrator": ["/content/piper_models/en_US-lessac-medium.onnx"],
        "male_adult": [
            "/content/piper_models/en_US-danny-medium.onnx",
            "/content/piper_models/en_US-hfc_male-medium.onnx",
            "/content/piper_models/en_GB-alan-medium.onnx"
        ],
        "female_adult": [
            "/content/piper_models/en_US-amy-medium.onnx",
            "/content/piper_models/en_GB-jenny_dioco-medium.onnx",
            "/content/piper_models/en_GB-alba-medium.onncolabnx"
        ],
        "male_teen": [
            "/content/piper_models/en_US-bryce-medium.onnx",
            "/content/piper_models/en_GB-northern_english_male-medium.onnx"
        ],
        "female_teen": [
            "/content/piper_models/en_US-hfc_female-medium.onnx",
            "/content/piper_models/en_GB-cori-medium.onnx"
        ],
        "male_child": ["/content/piper_models/en_US-bryce-medium.onnx"],
        "female_child": ["/content/piper_models/en_US-hfc_female-medium.onnx"],
        "male_elderly": ["/content/piper_models/en_US-arctic-medium.onnx"],
        "female_elderly": ["/content/piper_models/en_GB-aru-medium.onnx"],
        "unknown": ["/content/piper_models/en_US-lessac-medium.onnx"]
    }

    category = f"{gender}_{age_group}"
    # Fallback to unknown if category doesn't exist, then to narrator if unknown is empty
    pool = voice_pools.get(category, voice_pools.get("unknown", voice_pools["narrator"]))

    # Ensure pool is not empty
    if not pool:
        pool = voice_pools["narrator"]

    # Try to find a unique voice first
    for voice in pool:
        if voice not in used_voices.values():
            return voice

    # **IMPROVEMENT**: Rotate through the pool if all unique voices are used
    # This prevents defaulting to pool[0] every time
    used_count = sum(1 for v in used_voices.values() if v in pool)
    return pool[used_count % len(pool)]

def build_segments_from_dialogues(text, all_dialogues):
    """Build segments with narration and dialogue."""
    segments = []
    last_pos = 0

    # Sort all dialogues by position in text
    dialogue_positions = []
    for dialogue in all_dialogues:
        line = dialogue["line"]
        pos = text.find(line, last_pos)
        if pos != -1:
            dialogue_positions.append({
                "start": pos,
                "end": pos + len(line),
                "speaker_id": dialogue["speaker_id"],
                "line": line
            })

    dialogue_positions.sort(key=lambda x: x["start"])

    # Build segments
    last_pos = 0
    for dialogue_info in dialogue_positions:
        # Narration before dialogue
        if last_pos < dialogue_info["start"]:
            narration = text[last_pos:dialogue_info["start"]].strip()
            if narration:
                segments.append({
                    "speaker_id": "narrator",
                    "text": narration
                })

        # Dialogue
        segments.append({
            "speaker_id": dialogue_info["speaker_id"],
            "text": dialogue_info["line"]
        })

        last_pos = dialogue_info["end"]

    # Remaining narration
    if last_pos < len(text):
        narration = text[last_pos:].strip()
        if narration:
            segments.append({
                "speaker_id": "narrator",
                "text": narration
            })

    return segments

def generate_segment_audio(text, voice_model, output_file):
    """Generate audio for a single segment."""
    try:
        temp_text_file = f"{output_file}.txt"
        with open(temp_text_file, 'w', encoding='utf-8') as f:
            f.write(text)

        temp_output = f"{output_file}.temp.wav"
        piper_cmd = ["piper", "--model", voice_model, "--output_file", temp_output]

        with open(temp_text_file, 'r', encoding='utf-8') as f:
            result = subprocess.run(piper_cmd, stdin=f, capture_output=True, text=True)

        if os.path.exists(temp_text_file):
            os.remove(temp_text_file)

        if result.returncode != 0:
            return False

        # Normalize audio
        normalize_cmd = [
            "ffmpeg", "-i", temp_output,
            "-ar", "22050", "-ac", "1", "-sample_fmt", "s16",
            "-y", output_file
        ]

        result = subprocess.run(normalize_cmd, capture_output=True, text=True)

        if os.path.exists(temp_output):
            os.remove(temp_output)

        return result.returncode == 0 and os.path.exists(output_file)

    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        return False

def concatenate_audio_segments(segment_files, output_file):
    """Concatenate audio segments."""
    try:
        concat_list = f"{output_file}.concat.txt"
        with open(concat_list, 'w', encoding='utf-8') as f:
            for seg_file in segment_files:
                if not os.path.exists(seg_file):
                    return False
                f.write(f"file '{os.path.abspath(seg_file)}'\n")

        ffmpeg_cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-ar", "22050", "-ac", "1", "-sample_fmt", "s16",
            "-y", output_file
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if os.path.exists(concat_list):
            os.remove(concat_list)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Concatenation error: {e}")
        return False

def parallel_analyze_and_synthesize(text, work_dir):
    """
    Parallel pipeline: Analyze chunks with LLM while synthesizing previous results.
    Returns: (speaker_metadata, segments, multi_speaker_mode)
    """
    print("\n🚀 Starting parallel analysis & synthesis pipeline...")

    # Initialize character registry
    registry = CharacterRegistry()

    # Chunk the text intelligently
    chunks = smart_chunk_text(text, max_chunk_size=6000, overlap=300)
    print(f"   📚 Split into {len(chunks)} chunks (with 300-char overlap for context)")

    # Phase 1: Parallel LLM Analysis
    print("\n🧠 Phase 1: Parallel LLM Analysis")
    all_dialogues = []

    # Use ThreadPoolExecutor for parallel LLM calls
    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 to avoid VRAM issues
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(
                analyze_chunk_with_llm,
                chunk['text'],
                i,
                len(chunks),
                registry
            )
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            characters, dialogues, chunk_idx = future.result()
            all_dialogues.extend(dialogues)

    print(f"\n   ✅ Analysis complete: {len(registry.characters)} total characters, {len(all_dialogues)} dialogues")

    # Add narrator
    all_speakers = [{"id": "narrator", "name": "Narrator", "gender": "neutral", "age_group": "adult"}]
    all_speakers.extend(registry.get_all_characters())

    # Assign voices
    used_voices = {}
    speaker_metadata = {}

    print("\n🎤 Assigning voices:")
    for speaker in all_speakers:
        speaker_id = speaker['id']
        voice_model = assign_voice_model(
            speaker.get('gender', 'unknown'),
            speaker.get('age_group', 'adult'),
            speaker_id,
            used_voices
        )
        used_voices[speaker_id] = voice_model

        speaker_metadata[speaker_id] = {
            "name": speaker.get('name', speaker_id),
            "gender": speaker.get('gender', 'unknown'),
            "age": speaker.get('age_group', 'adult'),
            "voice_model": voice_model
        }

        print(f"   {speaker.get('name')}: {speaker.get('gender')} {speaker.get('age_group')} → {os.path.basename(voice_model)}")

    # Build segments
    segments = build_segments_from_dialogues(text, all_dialogues)
    multi_speaker_mode = len(all_dialogues) > 0

    print(f"\n   📝 Built {len(segments)} segments")

    return speaker_metadata, segments, multi_speaker_mode

# ========================================
# Job Management for Async Processing
# ========================================

import threading
from concurrent.futures import ThreadPoolExecutor
import uuid as uuid_lib

# Global job storage (in production, use Redis or database)
active_jobs = {}
completed_jobs = {}
job_lock = threading.Lock()

def process_conversion_async(job_id, clean_text, book_title):
    """Process conversion in background thread"""
    try:
        print(f"\n🚀 Starting async conversion for job {job_id}")

        # Update job status
        with job_lock:
            active_jobs[job_id]['status'] = 'processing'
            active_jobs[job_id]['progress'] = 'Converting text to audio...'

        # Setup paths
        job_dir = f"{WORK_DIR}/job_{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        text_file = f"{job_dir}/input.txt"
        audio_file = f"{job_dir}/final_audio.wav"
        mp3_file = f"{job_dir}/final_audio.mp3"
        timestamps_file = f"{job_dir}/timestamps.json"
        zip_file = f"{job_dir}/converted_book.zip"

        # Cleanup existing files
        for f in [text_file, audio_file, mp3_file, timestamps_file, zip_file]:
            if os.path.exists(f):
                os.remove(f)

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        # Update progress
        with job_lock:
            active_jobs[job_id]['progress'] = 'Analyzing text with LLM...'

        # Run the same conversion logic as before
        try:
            speaker_metadata, segments, multi_speaker_mode = parallel_analyze_and_synthesize(
                clean_text, job_dir
            )

            speakers_metadata_json = {
                "mode": "multi_voice" if multi_speaker_mode else "single_voice",
                "total_speakers": len(speaker_metadata),
                "llm_powered": True,
                "llm_model": "Qwen/Qwen2.5-7B-Instruct",
                "pipeline": "parallel",
                "speakers": [
                    {
                        "id": sid,
                        "name": meta["name"],
                        "gender": meta["gender"],
                        "age": meta["age"],
                        "voice_model": os.path.basename(meta["voice_model"])
                    }
                    for sid, meta in speaker_metadata.items()
                ],
                "segments_count": len(segments)
            }

        except Exception as e:
            print(f"   ⚠️ Pipeline failed for job {job_id}: {e}")
            # Fallback to single voice
            multi_speaker_mode = False
            segments = [{"speaker_id": "narrator", "text": clean_text}]
            speaker_metadata = {
                "narrator": {
                    "name": "Narrator",
                    "gender": "neutral",
                    "age": "adult",
                    "voice_model": PIPER_MODEL
                }
            }
            speakers_metadata_json = {
                "mode": "single_voice",
                "reason": f"Pipeline failed: {str(e)}",
                "llm_powered": False,
                "total_speakers": 1,
                "speakers": [{
                    "id": "narrator",
                    "name": "Narrator",
                    "gender": "neutral",
                    "age": "adult",
                    "voice_model": "en_US-lessac-medium"
                }],
                "segments_count": 1
            }

        # Update progress
        with job_lock:
            active_jobs[job_id]['progress'] = 'Generating audio...'

        # Generate audio (same logic as before)
        if multi_speaker_mode:
            print(f"\n🎵 Generating multi-voice audio ({len(segments)} segments)...")
            segment_audio_files = []

            for i, segment in enumerate(segments):
                speaker_id = segment['speaker_id']
                segment_text = segment['text']

                if not segment_text.strip():
                    continue

                voice_model = speaker_metadata[speaker_id]['voice_model']
                segment_audio_file = f"{job_dir}/segment_{i:04d}.wav"

                if i % 10 == 0:
                    with job_lock:
                        active_jobs[job_id]['progress'] = f'Generating audio segment {i}/{len(segments)}...'

                success = generate_segment_audio(segment_text, voice_model, segment_audio_file)

                if not success:
                    success = generate_segment_audio(segment_text, PIPER_MODEL, segment_audio_file)

                    if not success:
                        raise Exception(f"Segment {i} generation failed")

                segment_audio_files.append(segment_audio_file)

            print("\n🔗 Concatenating all segments...")
            with job_lock:
                active_jobs[job_id]['progress'] = 'Combining audio segments...'

            success = concatenate_audio_segments(segment_audio_files, audio_file)

            if not success:
                raise Exception("Audio concatenation failed")

            # Cleanup segment files
            for seg_file in segment_audio_files:
                if os.path.exists(seg_file):
                    os.remove(seg_file)

        else:
            print("\n🎵 Generating single-voice audio...")
            piper_cmd = f"piper --model {PIPER_MODEL} --output_file {audio_file} < {text_file}"
            result = subprocess.run(piper_cmd, shell=True, capture_output=True)

            if result.returncode != 0:
                raise Exception("TTS generation failed")

        # Convert to MP3
        with job_lock:
            active_jobs[job_id]['progress'] = 'Converting to MP3...'

        subprocess.run(
            f"ffmpeg -i {audio_file} -codec:a libmp3lame -qscale:a 2 {mp3_file} -y",
            shell=True, capture_output=True
        )

        if not os.path.exists(mp3_file):
            raise Exception("MP3 conversion failed")

        # Generate timestamps
        with job_lock:
            active_jobs[job_id]['progress'] = 'Generating timestamps...'

        try:
            audio_whisper = whisper.load_audio(mp3_file)
            whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
            result_whisper = whisper.transcribe(whisper_model, audio_whisper, language="en")

            timestamps = []
            for segment in result_whisper.get('segments', []):
                for word_info in segment.get('words', []):
                    timestamps.append({
                        "word": word_info.get('text', '').strip(),
                        "start": round(word_info.get('start', 0.0), 3),
                        "end": round(word_info.get('end', 0.0), 3)
                    })

            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps, f, indent=2)

        except Exception as e:
            print(f"⚠️ Timestamps failed for job {job_id}: {e}")

        # Package files
        with job_lock:
            active_jobs[job_id]['progress'] = 'Packaging files...'

        speakers_metadata_file = f"{job_dir}/speakers_metadata.json"
        with open(speakers_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(speakers_metadata_json, f, indent=2)

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(mp3_file, 'final_audio.mp3')
            if os.path.exists(timestamps_file):
                zipf.write(timestamps_file, 'timestamps.json')
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
        print(f"❌ Async conversion failed for job {job_id}: {e}")
        with job_lock:
            job_data = active_jobs.pop(job_id, {})
            completed_jobs[job_id] = {
                'status': 'failed',
                'error': str(e),
                'title': book_title,
                'completed_at': time.time(),
                'started_at': job_data.get('started_at', time.time())
            }

# Thread pool for async processing
conversion_executor = ThreadPoolExecutor(max_workers=2)

# ========================================
# Flask Endpoints
# ========================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "message": "TTS Server with Parallel LLM Pipeline",
        "gpu_available": torch.cuda.is_available(),
        "llm_loaded": model is not None,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "features": ["parallel_analysis", "unlimited_text_length", "speaker_consistency"]
    })

# New async endpoints
@app.route('/convert-async', methods=['POST'])
def convert_text_to_audio_async():
    """Start async conversion and return job ID immediately"""
    if not os.path.exists(PIPER_MODEL):
        return jsonify({"error": "Piper models missing"}), 503

    try:
        print("\n" + "="*50)
        print("📝 ASYNC CONVERSION REQUEST")
        print("="*50)

        # Parse request
        data = request.get_json(silent=True)
        if not data:
            raw = request.get_data(cache=False, as_text=True)
            if raw:
                try:
                    data = json.loads(raw)
                except Exception as e:
                    print(f"   JSON parse from raw failed: {e}")
                    data = None
        if (not data) and request.form:
            data = request.form.to_dict()

        if not data or 'text' not in data:
            print("   No 'text' in request body")
            return jsonify({"error": "No text provided"}), 400

        clean_text = data['text']
        book_title = data.get('title', 'book')

        print(f"📖 Book: {book_title}")
        print(f"📊 Length: {len(clean_text)} characters")

        #if len(clean_text) > 500000:
            #return jsonify({"error": "Text too long. Max 500k chars"}), 400

        # Generate job ID
        job_id = str(uuid_lib.uuid4())

        # Store job info
        with job_lock:
            active_jobs[job_id] = {
                'status': 'queued',
                'progress': 'Queued for processing...',
                'title': book_title,
                'text_length': len(clean_text),
                'started_at': time.time()
            }

        # Start async processing
        conversion_executor.submit(process_conversion_async, job_id, clean_text, book_title)

        print(f"✅ Job {job_id} queued for async processing")

        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Conversion started. Use /status/<job_id> to check progress."
        })

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of async conversion job"""
    try:
        with job_lock:
            # Check active jobs
            if job_id in active_jobs:
                job_data = active_jobs[job_id]
                return jsonify({
                    "job_id": job_id,
                    "status": job_data['status'],
                    "progress": job_data['progress'],
                    "title": job_data['title'],
                    "started_at": job_data['started_at'],
                    "elapsed_time": time.time() - job_data['started_at']
                })

            # Check completed jobs
            if job_id in completed_jobs:
                job_data = completed_jobs[job_id]
                response = {
                    "job_id": job_id,
                    "status": job_data['status'],
                    "title": job_data['title'],
                    "started_at": job_data['started_at'],
                    "completed_at": job_data['completed_at'],
                    "total_time": job_data['completed_at'] - job_data['started_at']
                }

                if job_data['status'] == 'failed':
                    response['error'] = job_data['error']
                else:
                    response['download_url'] = f"/download/{job_id}"

                return jsonify(response)

        return jsonify({"error": "Job not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<job_id>', methods=['GET'])
def download_converted_file(job_id):
    """Download completed conversion file"""
    try:
        with job_lock:
            if job_id not in completed_jobs:
                return jsonify({"error": "Job not found or not completed"}), 404

            job_data = completed_jobs[job_id]

            if job_data['status'] != 'completed':
                return jsonify({"error": "Job not completed successfully"}), 400

            zip_file = job_data['zip_file']

            if not os.path.exists(zip_file):
                return jsonify({"error": "Converted file not found"}), 404

            title = job_data['title']

            return send_file(
                zip_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"{title.replace(' ', '_')}_converted.zip"
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_text_to_audio():
    if not os.path.exists(PIPER_MODEL):
        return jsonify({"error": "Piper models missing"}), 503

    try:
        print("\n" + "="*50)
        print("📝 CONVERSION REQUEST (Parallel Pipeline)")
        print("="*50)

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        clean_text = data['text']
        book_title = data.get('title', 'book')

        print(f"📖 Book: {book_title}")
        print(f"📊 Length: {len(clean_text)} characters")

        if len(clean_text) > 500000:
            return jsonify({"error": "Text too long. Max 500k chars"}), 400

        # Setup paths
        text_file = f"{WORK_DIR}/input.txt"
        audio_file = f"{WORK_DIR}/final_audio.wav"
        mp3_file = f"{WORK_DIR}/final_audio.mp3"
        timestamps_file = f"{WORK_DIR}/timestamps.json"
        zip_file = f"{WORK_DIR}/converted_book.zip"

        # Cleanup
        for f in [text_file, audio_file, mp3_file, timestamps_file, zip_file]:
            if os.path.exists(f):
                os.remove(f)

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        # ===================================
        # PHASE: Parallel Analysis & Synthesis
        # ===================================

        try:
            speaker_metadata, segments, multi_speaker_mode = parallel_analyze_and_synthesize(
                clean_text,
                WORK_DIR
            )

            speakers_metadata_json = {
                "mode": "multi_voice" if multi_speaker_mode else "single_voice",
                "total_speakers": len(speaker_metadata),
                "llm_powered": True,
                "llm_model": "Qwen/Qwen2.5-7B-Instruct",
                "pipeline": "parallel",
                "speakers": [
                    {
                        "id": sid,
                        "name": meta["name"],
                        "gender": meta["gender"],
                        "age": meta["age"],
                        "voice_model": os.path.basename(meta["voice_model"])
                    }
                    for sid, meta in speaker_metadata.items()
                ],
                "segments_count": len(segments)
            }

        except Exception as e:
            print(f"   ⚠️ Pipeline failed: {e}")
            traceback.print_exc()

            # Fallback
            multi_speaker_mode = False
            segments = [{"speaker_id": "narrator", "text": clean_text}]
            speaker_metadata = {
                "narrator": {
                    "name": "Narrator",
                    "gender": "neutral",
                    "age": "adult",
                    "voice_model": PIPER_MODEL
                }
            }
            speakers_metadata_json = {
                "mode": "single_voice",
                "reason": f"Pipeline failed: {str(e)}",
                "llm_powered": False,
                "total_speakers": 1,
                "speakers": [{
                    "id": "narrator",
                    "name": "Narrator",
                    "gender": "neutral",
                    "age": "adult",
                    "voice_model": "en_US-lessac-medium"
                }],
                "segments_count": 1
            }

        # ===================================
        # PHASE: Audio Generation
        # ===================================

        if multi_speaker_mode:
            print(f"\n🎵 Generating multi-voice audio ({len(segments)} segments)...")
            segment_audio_files = []

            for i, segment in enumerate(segments):
                speaker_id = segment['speaker_id']
                segment_text = segment['text']

                if not segment_text.strip():
                    continue

                voice_model = speaker_metadata[speaker_id]['voice_model']
                segment_audio_file = f"{WORK_DIR}/segment_{i:04d}.wav"

                if i % 10 == 0:  # Progress update every 10 segments
                    print(f"   Progress: {i}/{len(segments)} segments")

                success = generate_segment_audio(segment_text, voice_model, segment_audio_file)

                if not success:
                    success = generate_segment_audio(segment_text, PIPER_MODEL, segment_audio_file)

                    if not success:
                        for temp in segment_audio_files:
                            if os.path.exists(temp):
                                os.remove(temp)
                        return jsonify({"error": f"Segment {i} generation failed"}), 500

                segment_audio_files.append(segment_audio_file)

            print("\n🔗 Concatenating all segments...")
            success = concatenate_audio_segments(segment_audio_files, audio_file)

            if not success:
                return jsonify({"error": "Concatenation failed"}), 500

            # Cleanup
            for seg_file in segment_audio_files:
                if os.path.exists(seg_file):
                    os.remove(seg_file)

            print("✅ Multi-voice audio complete")

        else:
            print("\n🎵 Generating single-voice audio...")
            piper_cmd = f"piper --model {PIPER_MODEL} --output_file {audio_file} < {text_file}"
            result = subprocess.run(piper_cmd, shell=True, capture_output=True)

            if result.returncode != 0:
                return jsonify({"error": "TTS failed"}), 500

        # Convert to MP3
        print("🔄 Converting to MP3...")
        subprocess.run(
            f"ffmpeg -i {audio_file} -codec:a libmp3lame -qscale:a 2 {mp3_file} -y",
            shell=True, capture_output=True
        )

        if not os.path.exists(mp3_file):
            return jsonify({"error": "MP3 conversion failed"}), 500

        # Generate timestamps
        print("⏱️ Generating timestamps...")
        try:
            audio_whisper = whisper.load_audio(mp3_file)
            whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
            result_whisper = whisper.transcribe(whisper_model, audio_whisper, language="en")

            timestamps = []
            for segment in result_whisper.get('segments', []):
                for word_info in segment.get('words', []):
                    timestamps.append({
                        "word": word_info.get('text', '').strip(),
                        "start": round(word_info.get('start', 0.0), 3),
                        "end": round(word_info.get('end', 0.0), 3)
                    })

            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps, f, indent=2)

            print(f"✅ Timestamps: {len(timestamps)} words")
        except Exception as e:
            print(f"⚠️ Timestamps failed: {e}")

        # Package
        print("📦 Packaging...")
        speakers_metadata_file = f"{WORK_DIR}/speakers_metadata.json"
        with open(speakers_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(speakers_metadata_json, f, indent=2)

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(mp3_file, 'final_audio.mp3')
            if os.path.exists(timestamps_file):
                zipf.write(timestamps_file, 'timestamps.json')
            zipf.write(text_file, 'book_text.txt')
            zipf.write(speakers_metadata_file, 'speakers_metadata.json')

        print("\n" + "="*50)
        print("✨ CONVERSION COMPLETE (Parallel Pipeline)")
        print("="*50)

        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{book_title.replace(' ', '_')}_converted.zip"
        )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========================================
# Streaming Endpoint
# ========================================

CHUNK_DELIMITER = b"--CHUNK_BOUNDARY--"
NEWLINE = b"\n"

@app.route("/stream", methods=["POST"])
def stream():
    if not os.path.exists(PIPER_MODEL):
        return jsonify({"error": "Piper models missing"}), 503

    try:
        print("\n🎧 Starting stream")
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text"}), 400

        def split_chunks(text, max_chars=1200):
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, current = [], ""
            for s in sentences:
                if len(current) + len(s) > max_chars:
                    chunks.append(current.strip())
                    current = ""
                current += s + " "
            if current.strip():
                chunks.append(current.strip())
            return chunks

        chunks = split_chunks(text)

        def generate():
            sample_rate = 22050
            bytes_per_sample = 2
            bytes_per_second = sample_rate * bytes_per_sample
            cumulative_duration = 0.0

            yield CHUNK_DELIMITER
            yield NEWLINE
            yield json.dumps({
                "type": "init",
                "session_id": str(uuid.uuid4()),
                "total_chunks": len(chunks)
            }).encode("utf-8")
            yield NEWLINE

            for i, chunk in enumerate(chunks, 1):
                piper_cmd = ["piper", "--model", PIPER_MODEL, "--output_file", "-"]
                ffmpeg_cmd = ["ffmpeg", "-f", "wav", "-i", "pipe:0",
                             "-f", "s16le", "-ac", "1", "-ar", "22050", "pipe:1",
                             "-y", "-loglevel", "error"]

                piper_process = subprocess.Popen(
                    piper_cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, stdin=piper_process.stdout,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                piper_process.stdout.close()

                out, err = piper_process.communicate(input=chunk.encode('utf-8'))
                pcm_data, ffmpeg_err = ffmpeg_process.communicate()

                if piper_process.returncode != 0:
                    print(f"❌ TTS error chunk {i}")
                    continue

                chunk_audio_data = pcm_data
                current_duration = len(chunk_audio_data) / bytes_per_second if bytes_per_second > 0 else 0

                chunk_start = cumulative_duration
                metadata = {
                    "type": "chunk_metadata",
                    "chunk_index": i,
                    "start_time": round(chunk_start, 3),
                    "duration": round(current_duration, 3),
                    "text": chunk,
                    "audio_size": len(chunk_audio_data)
                }

                yield CHUNK_DELIMITER
                yield NEWLINE
                yield json.dumps(metadata).encode('utf-8')
                yield NEWLINE
                yield chunk_audio_data

                cumulative_duration += current_duration
                time.sleep(0.1)

            yield CHUNK_DELIMITER
            yield NEWLINE
            yield json.dumps({
                "type": "end_of_stream",
                "total_duration": round(cumulative_duration, 3),
                "total_chunks": len(chunks)
            }).encode('utf-8')
            yield NEWLINE

        return Response(stream_with_context(generate()), mimetype="application/octet-stream")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========================================
# STEP 5: Start Server and Tunnel
# ========================================

def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def start_cloudflare_tunnel():
    """Start Cloudflare tunnel and return the process and URL"""
    print("\n🌐 Starting Cloudflare Tunnel...")

    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("❌ Cloudflare executable not found")
        print("Installing cloudflared...")
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb", shell=True)
        subprocess.run("dpkg -i cloudflared-linux-amd64.deb", shell=True)
        if not os.path.exists("/usr/local/bin/cloudflared"):
            return None, None

    tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://localhost:5001", "--metrics", "localhost:4040"]
    tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("⏳ Waiting for Cloudflare tunnel...")
    time.sleep(5)

    public_url = None
    for i in range(5):
        try:
            response = requests.get("http://localhost:4040/quicktunnel")
            response.raise_for_status()
            data = response.json()
            public_url = data['hostname']
            if public_url:
                public_url = f"https://{public_url}"
            break
        except:
            print(f"   ...retrying ({i+1}/5)")
            time.sleep(2)

    if public_url:
        return tunnel_process, public_url
    else:
        if tunnel_process:
            tunnel_process.kill()
        return None, None

def start_playit_tunnel():
    """Start playit.gg tunnel and return the process and URL"""
    print("\n🎮 Starting playit.gg Tunnel...")

    # Download playit if not present
    playit_path = "/usr/local/bin/playit"

    if os.path.exists(playit_path):
        print("✅ Playit already installed, skipping download")
    else:
        print("Installing playit...")
        subprocess.run("wget -O /usr/local/bin/playit https://github.com/playit-cloud/playit-agent/releases/latest/download/playit-linux-x86_64", shell=True)
        subprocess.run("chmod +x /usr/local/bin/playit", shell=True)
        if not os.path.exists(playit_path):
            print("❌ Failed to install playit")
            return None, None
        print("✅ Playit installed successfully")

    # Start playit tunnel
    tunnel_cmd = [playit_path]
    tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("⏳ Starting playit agent...")
    time.sleep(3)

    print("\n" + "="*70)
    print("📝 PLAYIT.GG SETUP INSTRUCTIONS:")
    print("="*70)
    print("1. The playit agent is running")
    print("2. Check the output above for a claim URL")
    print("3. Visit the URL to claim your agent")
    print("4. Create a TCP tunnel for port 5001")
    print("5. Get your tunnel URL (e.g., xxxxx.playit.gg:12345)")
    print("6. Use http://xxxxx.playit.gg:12345 in your Android app")
    print("="*70)
    print("\n⚠️ NOTE: Playit requires manual configuration")

    # For playit, we can't automatically get the URL
    return tunnel_process, "CHECK_PLAYIT_DASHBOARD"

def start_tunnel_and_get_url(tunnel_choice="auto"):
    """Start tunnel based on choice: 'cloudflare', 'playit', or 'auto'"""

    tunnel_process = None
    public_url = None
    tunnel_type = "Unknown"

    if tunnel_choice == "auto":
        # Try Cloudflare first, then playit
        print("\n🔄 Auto-selecting tunnel provider...")
        tunnel_process, public_url = start_cloudflare_tunnel()
        if public_url and public_url != "CHECK_PLAYIT_DASHBOARD":
            tunnel_type = "Cloudflare"
        else:
            print("\n⚠️ Cloudflare failed, trying playit.gg...")
            tunnel_process, public_url = start_playit_tunnel()
            tunnel_type = "playit.gg"
    elif tunnel_choice == "cloudflare":
        tunnel_process, public_url = start_cloudflare_tunnel()
        tunnel_type = "Cloudflare"
    elif tunnel_choice == "playit":
        tunnel_process, public_url = start_playit_tunnel()
        tunnel_type = "playit.gg"
    else:
        print(f"❌ Invalid tunnel choice: {tunnel_choice}")
        return

    if public_url:
        print("\n" + "="*70)
        print(f"🎉 TTS SERVER IS LIVE via {tunnel_type}!")
        print("="*70)

        if public_url == "CHECK_PLAYIT_DASHBOARD":
            print("\n⚠️  IMPORTANT: Get your URL from playit.gg")
            print("   1. Check the claim URL printed above")
            print("   2. Sign in and claim your agent")
            print("   3. Create a TCP tunnel for port 5001")
            print("   4. Get your tunnel URL from the dashboard")
            print("   5. Use http://xxxxx.playit.gg:port in your app")
        else:
            print(f"\n⚠️  COPY THIS URL TO YOUR APP:")
            print(f"➡️   {public_url}")

        print("="*70 + "\n")
        print("✨ Server Features:")
        print("  ✅ Multi-speaker detection with LLM")
        print("  ✅ Real-time streaming via /stream")
        print("  ✅ Full conversion via /convert")
        print("  ✅ Raw PCM output for Android AudioTrack")
        print("  ✅ Word-level timestamps with Whisper")
        print("\n🌐 Tunnel Options:")
        print(f"  • Current: {tunnel_type}")
        print("  • Cloudflare: Zero-config, automatic URL")
        print("  • playit.gg: More stable, requires dashboard setup")
        print("\n📊 System Info:")
        print(f"  🤖 LLM: Qwen/Qwen2.5-7B-Instruct")
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"  💾 Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print("\n🔒 Keep this running to maintain the server!")
        print("="*70 + "\n")
    else:
        print("\n❌ TUNNEL FAILED")
        if tunnel_process:
            tunnel_process.kill()
        return

    if tunnel_process:
        try:
            tunnel_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            tunnel_process.kill()
            print("✅ Server stopped")

# Kill existing processes
print("\n🔪 Cleaning up port 5001...")
lsof_exists = subprocess.run("command -v lsof", shell=True, capture_output=True).returncode == 0
if lsof_exists:
    !lsof -t -i:5001 | xargs -r kill -9 || true
else:
    print("⚠️  lsof not found")

# Start server
print("\n🚀 Starting Flask server with Parallel Pipeline...")
print("="*70)
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
time.sleep(3)

# ========================================
# TUNNEL SELECTION
# ========================================
# Choose your tunnel provider:
# - "cloudflare" : Automatic URL generation, zero-config (recommended for Colab)
# - "playit"     : More stable, requires manual dashboard setup (good for VPS)
# - "auto"       : Try Cloudflare first, fallback to playit

TUNNEL_CHOICE = "auto"  # Change to "cloudflare" or "playit" as needed

# Start tunnel
start_tunnel_and_get_url(TUNNEL_CHOICE)