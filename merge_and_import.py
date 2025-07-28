
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import argparse
import subprocess
import tempfile
import shutil
import sys

# --- Core Functions ---

def log_status(callback, message):
    """Helper to send status updates to the UI or print to console."""
    print(message)
    if callback:
        callback(message)

def find_llama_cpp_path():
    """
    Tries to find the llama.cpp repository in common locations.
    Searches for the 'convert.py' script.
    """
    # Search in the parent directory of the current script's project
    current_project_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_project_dir, '..'))
    
    search_paths = [
        os.path.join(parent_dir, 'llama.cpp'),
        os.path.join(os.path.expanduser("~"), 'llama.cpp') # User's home directory
    ]
    
    for path in search_paths:
        convert_script_path = os.path.join(path, 'convert_hf_to_gguf.py')
        if os.path.exists(convert_script_path):
            log_status(None, f"Found llama.cpp at: {path}")
            return path
            
    return None

def run_command(command, callback):
    """Runs a shell command and streams its output."""
    log_status(callback, f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_status(callback, output.strip())
            
    return process.poll() # Return the exit code

def do_merge_and_import(adapter_dir, ollama_model_name, status_callback=None):
    """
    Main logic for merging, converting, and importing the model.
    """
    try:
        # --- 1. Find llama.cpp ---
        log_status(status_callback, "Step 1: Searching for llama.cpp repository...")
        llama_cpp_path = find_llama_cpp_path()
        if not llama_cpp_path:
            log_status(status_callback, "ERROR: Could not find the 'llama.cpp' repository.")
            log_status(status_callback, "Please clone it into your project's parent directory or your home folder from https://github.com/ggerganov/llama.cpp.git")
            return False
        convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')

        # --- 2. Load Base Model and Merge LoRA ---
        log_status(status_callback, "Step 2: Loading base model and merging LoRA adapter...")
        config = PeftConfig.from_pretrained(adapter_dir)
        base_model_name = config.base_model_name_or_path
        
        log_status(status_callback, f"Base model: {base_model_name}")
        # --- 3. Save Merged Model to a Temporary Directory ---
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sub-directory for offloading within the main temporary directory
            offload_dir = os.path.join(temp_dir, "offload_cache")
            os.makedirs(offload_dir, exist_ok=True) # Ensure the directory exists

            log_status(status_callback, f"Step 3: Saving merged model to temporary directory: {temp_dir}")
            
            log_status(status_callback, "Loading tokenizer and model (this may take a while)...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder=offload_dir # Use the created offload directory
            )
            
            log_status(status_callback, "基础模型加载成功。") # 移动到这里，确保加载成功才打印
            
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
            
            # Using f16 as a common default. For smaller models, q8_0 might be better.
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
            # We need to run the command from within the temp_dir so Ollama can find the GGUF file
            import_command = [
                'ollama', 'create', ollama_model_name, '-f', 'Modelfile'
            ]
            
            # Ollama command needs to be run where the Modelfile and GGUF are
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            exit_code = run_command(import_command, status_callback)
            
            os.chdir(original_cwd) # Change back to the original directory
            
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

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter, convert to GGUF, and import into Ollama.")
    parser.add_argument(
        "adapter_dir", 
        type=str, 
        help="Path to the directory containing the 'final_lora_adapter'."
    )
    parser.add_argument(
        "ollama_model_name", 
        type=str, 
        help="The name for the model in Ollama (e.g., 'my-custom-model:7b')."
    )
    
    args = parser.parse_args()

    # Find the final_lora_adapter directory if a parent is provided
    final_adapter_path = os.path.join(args.adapter_dir, "final_lora_adapter")
    if not os.path.exists(final_adapter_path):
        print(f"Error: 'final_lora_adapter' not found in '{args.adapter_dir}'")
        sys.exit(1)

    do_merge_and_import(final_adapter_path, args.ollama_model_name)
