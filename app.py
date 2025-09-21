#-----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#-----------------------------------------------------------------------------------------

from flask import Flask, request, jsonify
import soundfile as sf
import os
import time
from voxcpm import VoxCPM

app = Flask(__name__)

# Load the TTS model at startup - only once
print("Loading VoxCPM model... this may take a minute...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
print("VoxCPM model loaded successfully!")

@app.route("/")
def hello():
    return app.send_static_file("index.html")

@app.route("/generate-speech", methods=["POST"])
def generate_speech():
    # Get text from request
    text = request.json.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Generate a unique filename
        timestamp = int(time.time())
        filename = f"output_{timestamp}.wav"
        filepath = os.path.join("static", filename)
        
        # Save the input text for reference
        text_file = os.path.join("static", f"text_{timestamp}.txt")
        with open(text_file, "w") as f:
            f.write(text)
        
        # Generate speech using VoxCPM model - this is the time-intensive step
        wav = model.generate(
            text=text,
            prompt_wav_path=None,
            prompt_text=None,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=True
        )
        
        # Save the generated audio
        sf.write(filepath, wav, 16000)
        
        # Return success response with the filename
        return jsonify({"success": True, "filename": filename})
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Print debug information
    print("Starting Flask app...")
    print(f"Routes: {[str(rule) for rule in app.url_map.iter_rules()]}")
    # Run with threading enabled to handle multiple requests
    app.run(host="0.0.0.0", port=9000, threaded=True)
