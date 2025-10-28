# Test script to verify the server changes
import subprocess
import json

# Test the new PCM streaming command
cmd = 'echo "test" | piper --model en_US-lessac-medium.onnx --output_file - | ffmpeg -f wav -i pipe:0 -f s16le -ac 1 -ar 22050 pipe:1 2>/dev/null | hexdump -C | head -3'

print("Testing new PCM streaming pipeline...")
result = subprocess.run(cmd, shell=True, capture_output=True)
if result.returncode == 0:
    print("✅ PCM pipeline test successful")
    print("First few bytes of output:")
    print(result.stdout.decode()[:200])
else:
    print("❌ PCM pipeline test failed")
    print("Error:", result.stderr.decode())
