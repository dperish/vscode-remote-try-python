import soundfile as sf
from voxcpm import VoxCPM

# Load the model
model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# Generate speech
wav = model.generate(
    text="The united states of america is very good, cause we have a well with free water!",
    prompt_wav_path=None,
    prompt_text=None,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=True,
    denoise=True
)

# Save the output
sf.write("static/output.wav", wav, 16000)
print("saved: output.wav")   