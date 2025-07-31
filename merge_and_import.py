import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import argparse
import subprocess
import tempfile
import shutil
import sys
import json

CONFIG_FILE = "config.json"

def get_llama_cpp_path(status_callback=None):
    """
    Finds the llama.cpp path, with config priority and auto-detection fallback.
    """
    # 1. Try to load from config file first
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        config_path = config.get("llama_cpp_path")
        if config_path and os.path.exists(os.path.join(config_path, 'convert_hf_to_gguf.py')):
            if status_callback: log_status(status_callback, f"Found valid llama.cpp path in config: {config_path}")
            return config_path

    # 2. If config fails, auto-detect in common locations
    if status_callback: log_status(status_callback, "Config path not found or invalid. Trying to auto-detect llama.cpp...")
    current_project_dir = os.path.dirname(__file__)
    search_paths = [
        os.path.abspath(os.path.join(current_project_dir, '..', 'llama.cpp')),
        os.path.join(os.path.expanduser("~"), 'llama.cpp')
    ]
    
    for path in search_paths:
        if os.path.exists(os.path.join(path, 'convert_hf_to_gguf.py')):
            if status_callback: log_status(status_callback, f"Auto-detected llama.cpp at: {path}. Saving to config.json.")
            # Auto-save the found path to config for future use
            try:
                config_data = {"llama_cpp_path": path}
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config_data, f, indent=4)
            except Exception as e:
                if status_callback: log_status(status_callback, f"Warning: Could not save auto-detected path to config.json: {e}")
            return path
            
    return None

# ... (The rest of the file remains the same, but all calls to the old get_llama_cpp_path will now use the new logic) ...

def log_status(callback, message):
    """Helper to send status updates to the UI or print to console."""
    print(message)
    if callback:
        callback(message)

def run_command(command, callback, cwd=None):
    """Runs a shell command and streams its output."""
    log_status(callback, f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd # Set the working directory for the command
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_status(callback, output.strip())
            
    return process.poll() # Return the exit code

def convert_base_model_to_ollama(base_model_id, ollama_model_name, status_callback=None):
    """
    Downloads a base Hugging Face model, converts it to GGUF, and imports it into Ollama.
    """
    try:
        # --- 1. Get llama.cpp path from config ---
        log_status(status_callback, "Step 1: Finding llama.cpp path...")
        llama_cpp_path = get_llama_cpp_path(status_callback)
        if not llama_cpp_path:
            log_status(status_callback, "ERROR: Could not find the 'llama.cpp' repository.")
            log_status(status_callback, "Please set the correct path in the 'Settings' tab or place it in the project's parent directory.")
            return False
        convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')

        # --- 2. Use a temporary directory for all artifacts ---
        with tempfile.TemporaryDirectory() as temp_dir:
            log_status(status_callback, f"Step 2: Using temporary directory: {temp_dir}")
            hf_model_path = os.path.join(temp_dir, "hf_model")

            # --- 3. Download/Load model from Hugging Face and save it locally ---
            log_status(status_callback, f"Step 3: Downloading/loading model '{base_model_id}' from Hugging Face...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
                
                log_status(status_callback, "Saving model to temporary local path...")
                model.save_pretrained(hf_model_path)
                tokenizer.save_pretrained(hf_model_path)
                log_status(status_callback, "Model saved successfully.")
            except Exception as e:
                log_status(status_callback, f"ERROR: Failed to download or save model from Hugging Face: {e}")
                return False

            # --- 4. Convert to GGUF ---
            log_status(status_callback, f"Step 4: Converting base model '{base_model_id}' to GGUF format...")
            gguf_output_path = os.path.join(temp_dir, f"{ollama_model_name}.gguf")
            
            convert_command = [
                sys.executable, convert_script, hf_model_path,
                '--outfile', gguf_output_path,
                '--outtype', 'f16'
            ]
            
            exit_code = run_command(convert_command, status_callback)
            if exit_code != 0:
                log_status(status_callback, f"ERROR: GGUF conversion failed with exit code {exit_code}.")
                return False
            log_status(status_callback, "GGUF conversion successful.")

            # --- 5. Create Ollama Modelfile ---
            log_status(status_callback, "Step 5: Creating Ollama Modelfile...")
            modelfile_content = f"FROM {os.path.basename(gguf_output_path)}"
            modelfile_path = os.path.join(temp_dir, 'Modelfile')
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            log_status(status_callback, f"Modelfile created at: {modelfile_path}")

            # --- 6. Import into Ollama ---
            log_status(status_callback, f"Step 6: Importing model '{ollama_model_name}' into Ollama...")
            import_command = [
                'ollama', 'create', ollama_model_name, '-f', 'Modelfile'
            ]
            
            exit_code = run_command(import_command, status_callback, cwd=temp_dir)
            
            if exit_code != 0:
                log_status(status_callback, f"ERROR: Ollama import failed with exit code {exit_code}.")
                return False

        log_status(status_callback, f"SUCCESS: Model '{ollama_model_name}' has been successfully imported into Ollama!")
        return True

    except Exception as e:
        log_status(status_callback, f"An unexpected error occurred: {e}")
        import traceback
        log_status(status_callback, traceback.format_exc())
        return False

def do_merge_and_import(adapter_dir, ollama_model_name, status_callback=None):
    """
    Main logic for merging, converting, and importing the model.
    """
    try:
        # --- 1. Get llama.cpp path from config ---
        log_status(status_callback, "Step 1: Finding llama.cpp path...")
        llama_cpp_path = get_llama_cpp_path(status_callback)
        if not llama_cpp_path:
            log_status(status_callback, "ERROR: Could not find the 'llama.cpp' repository.")
            log_status(status_callback, "Please set the correct path in the 'Settings' tab or place it in the project's parent directory.")
            return False
        convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')

        # --- 2. Load Base Model and Merge LoRA ---
        log_status(status_callback, "Step 2: Loading base model and merging LoRA adapter...")
        config = PeftConfig.from_pretrained(adapter_dir)
        base_model_name = config.base_model_name_or_path
        
        log_status(status_callback, f"Base model: {base_model_name}")
        # --- 3. Save Merged Model to a Temporary Directory ---
        with tempfile.TemporaryDirectory() as temp_dir:
            offload_dir = os.path.join(temp_dir, "offload_cache")
            os.makedirs(offload_dir, exist_ok=True)

            log_status(status_callback, f"Step 3: Saving merged model to temporary directory: {temp_dir}")
            
            log_status(status_callback, "Loading tokenizer and model (this may take a while)...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder=offload_dir
            )
            
            log_status(status_callback, "基础模型加载成功。")
            
            log_status(status_callback, "Applying LoRA adapter and merging...")
            model_with_lora = PeftModel.from_pretrained(base_model, adapter_dir, device_map="auto", offload_folder=offload_dir)
            merged_model = model_with_lora.merge_and_unload()
            log_status(status_callback, "Merge complete.")

            merged_model_path = os.path.join(temp_dir, 'merged_model')
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            # --- 4. Convert to GGUF ---
            log_status(status_callback, "Step 4: Converting merged model to GGUF format...")
            gguf_output_path = os.path.join(temp_dir, f"{ollama_model_name}.gguf")
            
            convert_command = [
                sys.executable, convert_script, merged_model_path,
                '--outfile', gguf_output_path,
                '--outtype', 'f16'
            ]
            
            exit_code = run_command(convert_command, status_callback)
            if exit_code != 0:
                log_status(status_callback, f"ERROR: GGUF conversion failed with exit code {exit_code}.")
                return False
            log_status(status_callback, "GGUF conversion successful.")

            # --- 5. Create Ollama Modelfile ---
            log_status(status_callback, "Step 5: Creating Ollama Modelfile...")
            modelfile_content = f"FROM {os.path.basename(gguf_output_path)}"
            modelfile_path = os.path.join(temp_dir, 'Modelfile')
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            log_status(status_callback, f"Modelfile created at: {modelfile_path}")

            # --- 6. Import into Ollama ---
            log_status(status_callback, f"Step 6: Importing model '{ollama_model_name}' into Ollama...")
            import_command = [
                'ollama', 'create', ollama_model_name, '-f', 'Modelfile'
            ]
            
            exit_code = run_command(import_command, status_callback, cwd=temp_dir)
            
            if exit_code != 0:
                log_status(status_callback, f"ERROR: Ollama import failed with exit code {exit_code}.")
                return False

        log_status(status_callback, f"SUCCESS: Model '{ollama_model_name}' has been successfully imported into Ollama!")
        return True

    except Exception as e:
        log_status(status_callback, f"An unexpected error occurred: {e}")
        import traceback
        log_status(status_callback, traceback.format_exc())
        return False

# --- Command-Line Interface (for testing) ---
if __name__ == "__main__":
    # This part is now primarily for testing the backend functions
    # The main application is main_app.py
    print("This script is not meant to be run directly. Use main_app.py")
