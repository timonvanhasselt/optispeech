import os
import soundfile as sf
from ospeech.inference import OptiSpeechONNXModel

# Dependencies: soundfile, numpy, onnxruntime, piper-phonemize-fix

# Download voice models first
# Emily: https://huggingface.co/mush42/optispeech/resolve/main/onnx/lightspeech/en-us/emily-step_400k.onnx
# Mike: https://huggingface.co/mush42/optispeech/resolve/main/onnx/lightspeech/en-us/mike-step_305k.onnx

# --- CONFIGURATION ---
MODEL_PATH = "mike-step_305k.onnx" # 
OUTPUT_DIR = "output_audio"
TEXT = "This is a test of Mike."

# Settings
D_FACTOR = 1.0  # Duration/Speed (0.5 is very fast, 2.0 is very slow)
P_FACTOR = 0.8  # Pitch (Lower than 1.0 is deeper, higher is squeakier)
E_FACTOR = 2.0  # Energy/Volume (Higher is more forceful)

def generate_audio():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"--- Loading model: {MODEL_PATH} ---")
    model = OptiSpeechONNXModel.from_onnx_file_path(MODEL_PATH)
    
    try:
        print(f"--- Preparing text with factors (D={D_FACTOR}, P={P_FACTOR})... ---")
        
        # In version 1.4.0, factors are usually passed here 
        # so they are encapsulated within the 'InferenceInputs' object.
        model_inputs = model.prepare_input(
            TEXT, 
            d_factor=D_FACTOR, 
            p_factor=P_FACTOR, 
            e_factor=E_FACTOR
        )

        print("--- Synthesizing... ---")
        # Now we only need to pass the formatted inputs
        outputs = model.synthesise(model_inputs)

        for i, wav in enumerate(outputs):
            output_file = os.path.join(OUTPUT_DIR, f"mike_test_{i}.wav")
            sf.write(output_file, wav, model.sample_rate)
            print(f"Success! File saved: {output_file}")

    except TypeError as e:
        print(f"\nKeyword error: {e}")
        print("Attempting via positional preparation...")
        # If d_factor=... doesn't work, we try the exact function argument order
        model_inputs = model.prepare_input(TEXT, True, D_FACTOR, P_FACTOR, E_FACTOR)
        outputs = model.synthesise(model_inputs)
        
        for i, wav in enumerate(outputs):
            output_file = os.path.join(OUTPUT_DIR, f"mike_positional_test_{i}.wav")
            sf.write(output_file, wav, model.sample_rate)
            print(f"Success! File saved: {output_file}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    generate_audio()