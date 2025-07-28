import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import os
import json
from inference_core import load_model_and_tokenizer, generate_response

class InferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA 推理助手 v1.0")
        self.root.geometry("800x700")

        self.model = None
        self.tokenizer = None
        self.status_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.temperature_var = tk.DoubleVar(value=0.8)
        self.context_mode_var = tk.BooleanVar(value=True)
        self.history = []

        # --- Main Frame ---
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Model Selection
        self.model_frame = ttk.LabelFrame(self.main_frame, text="1. 选择并加载模型", padding="10")
        self.model_frame.pack(fill=tk.X, expand=False, pady=5)
        
        self.model_path_label = ttk.Label(self.model_frame, text="未选择模型目录", width=70, relief="sunken")
        self.model_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.browse_button = ttk.Button(self.model_frame, text="选择目录...", command=self.browse_model_dir)
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = ttk.Button(self.model_frame, text="加载模型", command=self.load_model_thread, state=tk.DISABLED)
        self.load_button.pack(side=tk.RIGHT, padx=5)

        self.status_label = ttk.Label(self.main_frame, text="状态: 请先选择一个模型目录")
        self.status_label.pack(fill=tk.X, expand=False, pady=(0, 10))

        # 2. Inference Section
        self.inference_frame = ttk.LabelFrame(self.main_frame, text="2. 推理", padding="10")
        self.inference_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Instruction
        self.inst_frame = ttk.Frame(self.inference_frame)
        self.inst_frame.pack(fill=tk.X, expand=False, pady=5)
        self.inst_label = ttk.Label(self.inst_frame, text="指令 (Instruction):")
        self.inst_label.pack(side=tk.LEFT, padx=(0,5))
        self.inst_entry = ttk.Entry(self.inst_frame)
        self.inst_entry.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.inst_warning_label = ttk.Label(self.inst_frame, text="ⓘ 建议使用模型训练时的原始指令", foreground="blue")
        self.inst_warning_label.pack(side=tk.RIGHT, padx=(5,0))

        # Input
        self.input_label = ttk.Label(self.inference_frame, text="输入 (Input):")
        self.input_label.pack(anchor="w")
        self.input_text = scrolledtext.ScrolledText(self.inference_frame, height=5, wrap=tk.WORD)
        self.input_text.pack(fill=tk.X, expand=False, pady=5)

        # Temperature Slider
        self.temp_frame = ttk.Frame(self.inference_frame)
        self.temp_frame.pack(fill=tk.X, expand=False, pady=5)
        self.temp_label = ttk.Label(self.temp_frame, text=f"温度 (Temperature): {self.temperature_var.get():.2f}")
        self.temp_label.pack(side=tk.LEFT, padx=(0,10))
        self.temp_slider = ttk.Scale(
            self.temp_frame, 
            from_=0.1, 
            to=2.0, 
            orient=tk.HORIZONTAL, 
            variable=self.temperature_var,
            command=lambda v: self.temp_label.config(text=f"温度 (Temperature): {float(v):.2f}")
        )
        self.temp_slider.pack(fill=tk.X, expand=True, side=tk.LEFT)

        # Context Mode Controls
        self.context_frame = ttk.Frame(self.inference_frame)
        self.context_frame.pack(fill=tk.X, expand=False, pady=(10, 5))
        
        self.context_check = ttk.Checkbutton(
            self.context_frame, 
            text="启用上下文模式 (保留对话记忆)", 
            variable=self.context_mode_var
        )
        self.context_check.pack(side=tk.LEFT)

        self.clear_history_button = ttk.Button(
            self.context_frame, 
            text="清空对话历史", 
            command=self.clear_history
        )
        self.clear_history_button.pack(side=tk.RIGHT)

        # Generate Button
        self.generate_button = ttk.Button(self.inference_frame, text="发送", command=self.generate_response_thread, state=tk.DISABLED)
        self.generate_button.pack(pady=10)

        # Output -> Becomes Dialogue History
        self.output_label = ttk.Label(self.inference_frame, text="对话记录:")
        self.output_label.pack(anchor="w")
        self.output_text = scrolledtext.ScrolledText(self.inference_frame, height=10, state='disabled', wrap=tk.WORD, background="#f0f0f0")
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.root.after(100, self.periodic_check)

    def clear_history(self):
        self.history = []
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        messagebox.showinfo("操作成功", "对话历史已清空。")

    def browse_model_dir(self):
        dir_path = filedialog.askdirectory(title="选择包含 final_lora_adapter 的目录")
        if dir_path:
            # 寻找 final_lora_adapter 子目录
            adapter_path = os.path.join(dir_path, "final_lora_adapter")
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                self.model_path_label.config(text=adapter_path)
                self.load_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"状态: 已选择模型, 请点击 '加载模型'")
                # 尝试自动加载指令
                self.auto_load_instruction(dir_path)
            else:
                messagebox.showerror("错误", f"在 '{dir_path}' 中未找到 'final_lora_adapter' 子目录或其配置文件。请选择正确的顶层输出目录。")
                self.clear_history() # Also clear history if model load fails

    def auto_load_instruction(self, base_dir):
        # 这是一个简单的尝试，寻找目录下的 .jsonl 文件并读取指令
        try:
            for file in os.listdir(base_dir):
                if file.endswith(".jsonl"):
                    with open(os.path.join(base_dir, file), 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        data = json.loads(first_line)
                        if 'instruction' in data:
                            self.inst_entry.delete(0, tk.END)
                            self.inst_entry.insert(0, data['instruction'])
                            self.status_label.config(text="状态: 已自动加载推荐指令, 请点击 '加载模型'")
                            return
        except Exception:
            # 静默失败，找不到就算了
            pass

    def load_model_thread(self):
        adapter_path = self.model_path_label.cget("text")
        self.load_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.DISABLED)
        threading.Thread(target=self.load_model, args=(adapter_path,), daemon=True).start()

    def load_model(self, adapter_path):
        self.model, self.tokenizer = load_model_and_tokenizer(adapter_path, self.status_queue)

    def generate_response_thread(self):
        instruction = self.inst_entry.get()
        input_text = self.input_text.get("1.0", tk.END).strip()
        temperature = self.temperature_var.get()
        if not input_text:
            messagebox.showwarning("警告", "请输入内容！")
            return
            
        self.generate_button.config(state=tk.DISABLED)
        
        # Append user message to the chat log immediately
        self.update_chat_log("You", input_text)
        self.input_text.delete("1.0", tk.END) # Clear input box
        
        # Prepare history for the model
        history_to_send = self.history if self.context_mode_var.get() else []
        
        threading.Thread(target=self.run_generation, args=(instruction, input_text, temperature, history_to_send), daemon=True).start()

    def run_generation(self, instruction, input_text, temperature, history):
        response = generate_response(self.model, self.tokenizer, instruction, input_text, history, temperature)
        # Put the original input and the response into the queue
        self.response_queue.put((input_text, response))

    def update_chat_log(self, role, text):
        self.output_text.config(state='normal')
        if role == "You":
            self.output_text.insert(tk.END, f"You:\n{text}\n\n", "user_tag")
        elif role == "Model":
            self.output_text.insert(tk.END, f"Model:\n{text}\n\n", "model_tag")
        
        self.output_text.config(state='disabled')
        self.output_text.yview(tk.END) # Auto-scroll to the bottom

    def periodic_check(self):
        # Check status queue from model loading
        while not self.status_queue.empty():
            status = self.status_queue.get_nowait()
            self.status_label.config(text=f"状态: {status}")
            if "错误" in status:
                self.load_button.config(state=tk.NORMAL)
                self.browse_button.config(state=tk.NORMAL)
            if "模型准备就绪" in status:
                self.generate_button.config(state=tk.NORMAL)
                self.browse_button.config(state=tk.NORMAL)
                # Configure tags for chat styling
                self.output_text.tag_config('user_tag', foreground='blue', font=('Arial', 10, 'bold'))
                self.output_text.tag_config('model_tag', foreground='darkgreen', font=('Arial', 10))


        # Check response queue from generation
        while not self.response_queue.empty():
            input_text, response = self.response_queue.get_nowait()
            
            # Update chat log with model's response
            self.update_chat_log("Model", response)
            
            # Update history if context mode is on
            if self.context_mode_var.get():
                self.history.append((input_text, response))
            
            self.generate_button.config(state=tk.NORMAL)

        self.root.after(100, self.periodic_check)

if __name__ == "__main__":
    root = tk.Tk()
    app = InferenceGUI(root)
    root.mainloop() 