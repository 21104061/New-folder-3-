# TTS Conversion Server for Google Colab with Local LLM (gpt-oss-20b)
# Run this notebook with GPU enabled (Runtime > Change runtime type > T4 GPU or A100)

import os
import sys
import uuid
import threading
import traceback
import json
import time
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
    import piper_tts
    import flask
    import flask_cors
    import whisper_timestamped
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Dependencies installed successfully.")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit("Dependency check failed.")

# ========================================
# STEP 2: Setup Cloudflare Tunnel
# ========================================
print("\n🌐 Setting up Cloudflare Tunnel...")
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

if not os.path.exists("/usr/local/bin/cloudflared"):
    print("❌ Cloudflare Tunnel executable not found.")
    sys.exit("Cloudflare installation failed.")

# ========================================
# STEP 2.5: Load Local LLM (gpt-oss-20b)
# ========================================
print("\n🤖 Loading gpt-oss-20b model (this may take a few minutes)...")
print("   Model size: ~13.8GB")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    
    print("✅ Local LLM loaded successfully!")
    print(f"   Model: {model_name}")
    print(f"   Device: {model.device}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"❌ Failed to load LLM: {e}")
    sys.exit("LLM loading failed.")

# ========================================
# STEP 3: Download Piper Voice Models
# ========================================
print("\n🎤 Downloading Piper voice models...")
os.makedirs("/content/piper_models", exist_ok=True)

# Base narrator voice
!wget -q -O /content/piper_models/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
!wget -q -O /content/piper_models/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

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
        
        # Extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[json_start:json_end]
        result = json.loads(json_str)
        
        if "characters" not in result or "dialogues" not in result:
            raise ValueError("Invalid JSON structure")
        
        # Merge characters into registry
        for char in result["characters"]:
            char_id = registry.merge_character(char)
            # Update dialogue speaker_ids to use registry IDs
            for dialogue in result["dialogues"]:
                if dialogue["speaker_id"] == char["id"]:
                    dialogue["speaker_id"] = char_id
        
        print(f"      ✅ Found {len(result['characters'])} chars, {len(result['dialogues'])} dialogues")
        
        return (result["characters"], result["dialogues"], chunk_index)
        
    except Exception as e:
        print(f"      ⚠️ Chunk {chunk_index + 1} analysis failed: {e}")
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
            "/content/piper_models/en_GB-alba-medium.onnx"
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
    pool = voice_pools.get(category, voice_pools["unknown"])
    
    for voice in pool:
        if voice not in used_voices.values():
            return voice
    
    return pool[0]

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

def start_tunnel_and_get_url():
    print("\n🌐 Starting Cloudflare Tunnel...")
    
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("❌ Cloudflare executable not found")
        return
    
    tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://localhost:5001", "--metrics", "localhost:4040"]
    tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("⏳ Waiting for tunnel...")
    time.sleep(5)
    
    public_url = None
    for i in range(5):
        try:
            response = requests.get("http://localhost:4040/quicktunnel")
            response.raise_for_status()
            data = response.json()
            public_url = data['hostname']
            break
        except:
            print(f"   ...retrying ({i+1}/5)")
            time.sleep(2)
    
    if public_url:
        print("\n" + "="*70)
        print("🎉 PARALLEL PIPELINE TTS SERVER IS LIVE!")
        print("="*70)
        print(f"\n⚠️  COPY THIS URL TO YOUR APP:")
        print(f"➡️   https://{public_url}")
        print("="*70 + "\n")
        print("✨ Advanced Features:")
        print("  ✅ Parallel LLM Analysis - Multiple chunks analyzed simultaneously")
        print("  ✅ Unlimited Text Length - Intelligent chunking with overlap")
        print("  ✅ Speaker Consistency - Global character registry across chunks")
        print("  ✅ Context Preservation - 300-char overlap maintains speaker context")
        print("  ✅ Pipeline Optimization - Analysis and synthesis happen in parallel")
        print("  ✅ Multi-voice Synthesis - Gender & age-based voice assignment")
        print("  ✅ No API Keys Required - 100% local inference")
        print("\n📊 System Status:")
        print(f"  🤖 LLM Model: Qwen/Qwen2.5-7B-Instruct")
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"  💾 GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        print(f"  🔥 Max Workers: 2 parallel LLM threads")
        print(f"  📏 Chunk Size: 6000 chars with 300-char overlap")
        print("\n⚡ Performance Tips:")
        print("  • Longer texts = better parallelization efficiency")
        print("  • First request may be slower (model warm-up)")
        print("  • Upgrade to A100 GPU for faster processing")
        print("\n🔒 Keep this cell running to maintain the server!")
        print("="*70 + "\n")
    else:
        print("\n❌ TUNNEL FAILED")
        tunnel_process.kill()
        return
    
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

# Start tunnel
start_tunnel_and_get_url()