import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, simpledialog
import threading
import queue
import time
import os
import json
from train_core import start_training, get_local_lora_base_models, get_existing_lora_dirs
from merge_and_import import do_merge_and_import

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA 微调与管理助手 v1.1")

        # --- Queues for threading ---
        self.progress_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.merge_log_queue = queue.Queue()

        # --- Main Frame ---
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- New LoRA Training Section ---
        self.new_lora_frame = ttk.LabelFrame(self.main_frame, text="新 LoRA 训练 (从头开始)", padding="10")
        self.new_lora_frame.pack(fill=tk.X, expand=False, pady=5)

        # 1. Model Selection (New LoRA)
        self.new_lora_model_selection_frame = ttk.Frame(self.new_lora_frame)
        self.new_lora_model_selection_frame.pack(fill=tk.X, expand=False, pady=5)
        self.new_lora_base_model_label = ttk.Label(self.new_lora_model_selection_frame, text="基座模型:")
        self.new_lora_base_model_label.pack(side=tk.LEFT, padx=(0, 5))
        self.new_lora_model_combobox = ttk.Combobox(self.new_lora_model_selection_frame, state="normal", width=60) # 可编辑
        self.new_lora_model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.new_lora_model_combobox.set("请选择或输入 Hugging Face 模型 ID...")
        self.new_lora_refresh_button = ttk.Button(self.new_lora_model_selection_frame, text="刷新", command=self._refresh_new_lora_models)
        self.new_lora_refresh_button.pack(side=tk.RIGHT, padx=5)

        # 2. Data Selection (New LoRA)
        self.new_lora_data_frame = ttk.Frame(self.new_lora_frame)
        self.new_lora_data_frame.pack(fill=tk.X, expand=False, pady=5)
        self.new_lora_data_path_label_desc = ttk.Label(self.new_lora_data_frame, text="训练数据:")
        self.new_lora_data_path_label_desc.pack(side=tk.LEFT, padx=(0, 5))
        self.new_lora_data_path_label = ttk.Label(self.new_lora_data_frame, text="未选择文件", width=70, relief="sunken")
        self.new_lora_data_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.new_lora_browse_button = ttk.Button(self.new_lora_data_frame, text="浏览...", command=self.browse_file_new_lora)
        self.new_lora_browse_button.pack(side=tk.RIGHT, padx=5)

        # 3. Output Directory (New LoRA)
        self.new_lora_output_frame = ttk.Frame(self.new_lora_frame)
        self.new_lora_output_frame.pack(fill=tk.X, expand=False, pady=5)
        self.new_lora_output_dir_label = ttk.Label(self.new_lora_output_frame, text="输出目录:")
        self.new_lora_output_dir_label.pack(side=tk.LEFT, padx=(0, 5))
        self.new_lora_output_dir_entry = ttk.Entry(self.new_lora_output_frame, width=70)
        self.new_lora_output_dir_entry.pack(fill=tk.X, expand=True, padx=5)
        self.new_lora_output_dir_entry.insert(0, "./lora_finetuned_model_")

        # 4. Control Button (New LoRA)
        self.new_lora_start_button = ttk.Button(self.new_lora_frame, text="开始新 LoRA 训练", command=self.start_new_lora_training_thread)
        self.new_lora_start_button.pack(pady=10)

        # --- Continue Training Existing LoRA Section ---
        self.continue_lora_frame = ttk.LabelFrame(self.main_frame, text="继续训练现有 LoRA", padding="10")
        self.continue_lora_frame.pack(fill=tk.X, expand=False, pady=5)

        # 1. Existing LoRA Selection
        self.existing_lora_selection_frame = ttk.Frame(self.continue_lora_frame)
        self.existing_lora_selection_frame.pack(fill=tk.X, expand=False, pady=5)
        self.existing_lora_label = ttk.Label(self.existing_lora_selection_frame, text="选择现有 LoRA:")
        self.existing_lora_label.pack(side=tk.LEFT, padx=(0, 5))
        self.existing_lora_combobox = ttk.Combobox(self.existing_lora_selection_frame, state="readonly", width=60)
        self.existing_lora_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.existing_lora_combobox.set("扫描本地已训练 LoRA 模型...")
        self.existing_lora_refresh_button = ttk.Button(self.existing_lora_selection_frame, text="刷新", command=self._refresh_existing_lora_models)
        self.existing_lora_refresh_button.pack(side=tk.RIGHT, padx=5)

        # 2. Data Selection (Continue LoRA)
        self.continue_lora_data_frame = ttk.Frame(self.continue_lora_frame)
        self.continue_lora_data_frame.pack(fill=tk.X, expand=False, pady=5)
        self.continue_lora_data_path_label_desc = ttk.Label(self.continue_lora_data_frame, text="训练数据:")
        self.continue_lora_data_path_label_desc.pack(side=tk.LEFT, padx=(0, 5))
        self.continue_lora_data_path_label = ttk.Label(self.continue_lora_data_frame, text="未选择文件", width=70, relief="sunken")
        self.continue_lora_data_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.continue_lora_browse_button = ttk.Button(self.continue_lora_data_frame, text="浏览...", command=self.browse_file_continue_lora)
        self.continue_lora_browse_button.pack(side=tk.RIGHT, padx=5)

        # 3. Output Directory (Continue LoRA)
        self.continue_lora_output_frame = ttk.Frame(self.continue_lora_frame)
        self.continue_lora_output_frame.pack(fill=tk.X, expand=False, pady=5)
        self.continue_lora_output_dir_label = ttk.Label(self.continue_lora_output_frame, text="输出目录:")
        self.continue_lora_output_dir_label.pack(side=tk.LEFT, padx=(0, 5))
        self.continue_lora_output_dir_entry = ttk.Entry(self.continue_lora_output_frame, width=70)
        self.continue_lora_output_dir_entry.pack(fill=tk.X, expand=True, padx=5)
        self.continue_lora_output_dir_entry.insert(0, "./lora_finetuned_model_continued_")

        # 4. Control Button (Continue LoRA)
        self.continue_lora_start_button = ttk.Button(self.continue_lora_frame, text="继续训练", command=self.start_continue_lora_training_thread)
        self.continue_lora_start_button.pack(pady=10)

        # --- Progress and Logs ---
        self.progress_frame = ttk.LabelFrame(self.main_frame, text="训练进度", padding="10")
        self.progress_frame.pack(fill=tk.X, expand=False, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=2)
        self.status_label = ttk.Label(self.progress_frame, text="状态: 空闲")
        self.status_label.pack(fill=tk.X, expand=True, padx=5, pady=2)

        # --- Model Management Section ---
        self.management_frame = ttk.LabelFrame(self.main_frame, text="模型管理", padding="10")
        self.management_frame.pack(fill=tk.X, expand=False, pady=10)

        # Merge and Import to Ollama
        self.merge_ollama_frame = ttk.Frame(self.management_frame)
        self.merge_ollama_frame.pack(fill=tk.X, expand=False, pady=5)
        self.merge_ollama_label = ttk.Label(self.merge_ollama_frame, text="选择要合并的模型:")
        self.merge_ollama_label.pack(side=tk.LEFT, padx=(0, 5))
        self.merge_ollama_combobox = ttk.Combobox(self.merge_ollama_frame, state="readonly", width=60)
        self.merge_ollama_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.merge_ollama_combobox.set("扫描本地已训练 LoRA 模型...")
        self.merge_ollama_refresh_button = ttk.Button(self.merge_ollama_frame, text="刷新", command=self._refresh_merge_models)
        self.merge_ollama_refresh_button.pack(side=tk.RIGHT, padx=5)

        self.merge_button = ttk.Button(self.management_frame, text="合并 LoRA 并导入到 Ollama", command=self.start_merge_and_import_thread)
        self.merge_button.pack(pady=5)

        # --- Log Viewer ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="实时日志", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, state='disabled', wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Internal State ---
        self.training_thread = None
        self.merge_thread = None
        self.selected_data_file_new_lora = ""
        self.selected_data_file_continue_lora = ""

        self._refresh_new_lora_models() # 初始加载新 LoRA 训练的模型列表
        self._refresh_existing_lora_models() # 初始加载现有 LoRA 训练的模型列表
        self._refresh_merge_models() # 初始加载合并模型的列表
        self.root.after(100, self.periodic_check)

    def _refresh_new_lora_models(self):
        self.new_lora_model_combobox.set("正在扫描本地 LoRA 基座模型...")
        self.new_lora_refresh_button.config(state=tk.DISABLED)
        models, error = get_local_lora_base_models() # 调用新的函数
        if error:
            messagebox.showerror("错误", error)
            self.new_lora_model_combobox.set("扫描失败")
            self.new_lora_model_combobox['values'] = []
        else:
            # 添加手动输入选项
            display_models = ["--手动输入 Hugging Face 模型 ID--"] + models
            self.new_lora_model_combobox['values'] = display_models
            if models:
                self.new_lora_model_combobox.set(models[0]) # 默认选择第一个识别到的模型
            else:
                self.new_lora_model_combobox.set(display_models[0]) # 如果没有模型，默认选择手动输入选项
        self.new_lora_refresh_button.config(state=tk.NORMAL)

    def _refresh_existing_lora_models(self):
        self.existing_lora_combobox.set("正在扫描本地已训练 LoRA 模型...")
        self.existing_lora_refresh_button.config(state=tk.DISABLED)
        models, error = get_existing_lora_dirs() # 调用新的函数
        if error:
            messagebox.showerror("错误", error)
            self.existing_lora_combobox.set("扫描失败")
            self.existing_lora_combobox['values'] = []
        elif not models:
            self.existing_lora_combobox.set("未找到本地已训练 LoRA 模型")
            self.existing_lora_combobox['values'] = []
        else:
            self.existing_lora_combobox['values'] = models
            self.existing_lora_combobox.set(models[0])
        self.existing_lora_refresh_button.config(state=tk.NORMAL)

    def _refresh_merge_models(self):
        self.merge_ollama_combobox.set("正在扫描本地已训练 LoRA 模型...")
        self.merge_ollama_refresh_button.config(state=tk.DISABLED)
        models, error = get_existing_lora_dirs() # 调用新的函数
        if error:
            messagebox.showerror("错误", error)
            self.merge_ollama_combobox.set("扫描失败")
            self.merge_ollama_combobox['values'] = []
        elif not models:
            self.merge_ollama_combobox.set("未找到本地已训练 LoRA 模型")
            self.merge_ollama_combobox['values'] = []
        else:
            self.merge_ollama_combobox['values'] = models
            self.merge_ollama_combobox.set(models[0])
        self.merge_ollama_refresh_button.config(state=tk.NORMAL)

    def browse_file_new_lora(self):
        file_path = filedialog.askopenfilename(
            title="选择数据集文件 (新 LoRA 训练)",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*" ))
        )
        if file_path:
            self.selected_data_file_new_lora = file_path
            self.new_lora_data_path_label.config(text=file_path)

            filename = os.path.basename(file_path).split('.')[0]
            self.new_lora_output_dir_entry.delete(0, tk.END)
            self.new_lora_output_dir_entry.insert(0, f"./lora_{filename}_finetuned")

    def browse_file_continue_lora(self):
        file_path = filedialog.askopenfilename(
            title="选择数据集文件 (继续训练现有 LoRA)",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*" ))
        )
        if file_path:
            self.selected_data_file_continue_lora = file_path
            self.continue_lora_data_path_label.config(text=file_path)

            filename = os.path.basename(file_path).split('.')[0]
            self.continue_lora_output_dir_entry.delete(0, tk.END)
            self.continue_lora_output_dir_entry.insert(0, f"./lora_{filename}_finetuned_continued")

    def start_new_lora_training_thread(self):
        if self.is_busy(): return
        
        hf_model_id = self.new_lora_model_combobox.get().strip()
        lora_adapter_path = None

        if hf_model_id == "--手动输入 Hugging Face 模型 ID--":
            hf_model_id = simpledialog.askstring("输入 Hugging Face 模型 ID", "请输入 Hugging Face 模型 ID (例如: Qwen/Qwen1.5-4B-Chat):")
            if not hf_model_id:
                messagebox.showwarning("警告", "未输入 Hugging Face 模型 ID，训练已取消。")
                return

        if not hf_model_id:
            messagebox.showerror("错误", "请选择或输入一个有效的 Hugging Face 模型 ID！")
            return

        if not self.selected_data_file_new_lora:
            messagebox.showerror("错误", "请先选择一个数据集文件！")
            return
        output_dir = self.new_lora_output_dir_entry.get()
        if not output_dir:
            messagebox.showerror("错误", "请输入一个输出目录名！")
            return

        self.set_ui_busy(True)
        self.clear_logs()
        
        messagebox.showinfo("模型路径提示", f"将基于 Hugging Face 模型 '{hf_model_id}' 从头开始训练新的 LoRA 适配器。")

        self.training_thread = threading.Thread(
            target=start_training,
            args=(hf_model_id, self.selected_data_file_new_lora, output_dir, self.progress_queue, self.log_queue, lora_adapter_path),
            daemon=True
        )
        self.training_thread.start()

    def start_continue_lora_training_thread(self):
        if self.is_busy(): return

        lora_base_dir = self.existing_lora_combobox.get().strip()
        if not lora_base_dir or "扫描失败" in lora_base_dir or "未找到" in lora_base_dir:
            messagebox.showerror("错误", "请先选择一个有效的本地已训练 LoRA 模型！")
            return
        
        lora_adapter_path = os.path.join(lora_base_dir, "final_lora_adapter")
        if not os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
            messagebox.showerror("错误", f"在 '{lora_base_dir}' 中未找到有效的 'final_lora_adapter' 子目录或其配置文件。")
            return

        # 从 adapter_config.json 中读取原始基座模型名称
        hf_model_id = None
        try:
            with open(os.path.join(lora_adapter_path, "adapter_config.json"), 'r') as f:
                config = json.load(f)
                hf_model_id = config.get("base_model_name_or_path")
            if not hf_model_id:
                messagebox.showerror("错误", f"无法从 {lora_adapter_path} 中的 adapter_config.json 获取原始基座模型名称。")
                return
        except Exception as e:
            messagebox.showerror("错误", f"读取 LoRA 适配器配置失败: {e}")
            return

        if not self.selected_data_file_continue_lora:
            messagebox.showerror("错误", "请先选择一个数据集文件！")
            return
        output_dir = self.continue_lora_output_dir_entry.get()
        if not output_dir:
            messagebox.showerror("错误", "请输入一个输出目录名！")
            return

        self.set_ui_busy(True)
        self.clear_logs()
        
        messagebox.showinfo("模型路径提示", f"将基于原始基座模型 '{hf_model_id}' 和现有 LoRA 适配器 '{lora_adapter_path}' 继续训练。")

        self.training_thread = threading.Thread(
            target=start_training,
            args=(hf_model_id, self.selected_data_file_continue_lora, output_dir, self.progress_queue, self.log_queue, lora_adapter_path),
            daemon=True
        )
        self.training_thread.start()

    def start_merge_and_import_thread(self):
        if self.is_busy(): return

        adapter_dir = self.merge_ollama_combobox.get().strip()
        if not adapter_dir or "扫描失败" in adapter_dir or "未找到" in adapter_dir:
            messagebox.showerror("错误", "请先选择一个有效的本地已训练 LoRA 模型进行合并！")
            return
        
        final_adapter_path = os.path.join(adapter_dir, "final_lora_adapter")
        if not os.path.exists(os.path.join(final_adapter_path, "adapter_config.json")):
            messagebox.showerror("错误", f"在 '{adapter_dir}' 中未找到有效的 'final_lora_adapter' 子目录或其配置文件。")
            return

        ollama_model_name = simpledialog.askstring("输入模型名称", "请输入您想在 Ollama 中使用的模型名称 (例如: my-model:7b-finetuned):")
        if not ollama_model_name:
            return

        self.set_ui_busy(True)
        self.clear_logs()
        self.status_label.config(text="状态: 正在合并与导入模型...")

        self.merge_thread = threading.Thread(
            target=do_merge_and_import,
            args=(final_adapter_path, ollama_model_name, self.merge_log_queue.put),
            daemon=True
        )
        self.merge_thread.start()

    def periodic_check(self):
        # Check training progress queue
        while not self.progress_queue.empty():
            data = self.progress_queue.get_nowait()
            
            if 'error' in data:
                messagebox.showerror("训练失败", f"发生错误: {data['error']}")
                self.set_ui_busy(False)
                return

            self.progress_bar['value'] = data['progress']
            loss = data.get('loss', 'N/A')
            eta_seconds = data.get('eta_seconds', float('inf'))
            
            eta_str = "计算中..."
            if eta_seconds != float('inf'):
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
            status_text = f"进度: {data['progress']:.2f}% | 当前 Loss: {loss} | 预计剩余时间: {eta_str}"
            self.status_label.config(text=status_text)
            
            if data.get('done'):
                messagebox.showinfo("成功", "训练已成功完成！")
                self.set_ui_busy(False)

        # Check training log queue
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get_nowait())

        # Check merge & import log queue
        while not self.merge_log_queue.empty():
            log_entry = self.merge_log_queue.get_nowait()
            self.append_log(log_entry)
            if "SUCCESS:" in log_entry:
                messagebox.showinfo("成功", log_entry)
                self.set_ui_busy(False)
                self.status_label.config(text="状态: 空闲")
            elif "ERROR:" in log_entry:
                messagebox.showerror("失败", log_entry)
                self.set_ui_busy(False)
                self.status_label.config(text="状态: 空闲")

        self.root.after(100, self.periodic_check)

    def set_ui_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        # New LoRA Training controls
        self.new_lora_model_combobox.config(state='readonly' if busy else 'normal')
        self.new_lora_refresh_button.config(state=state)
        self.new_lora_data_path_label.config(state=state)
        self.new_lora_browse_button.config(state=state)
        self.new_lora_output_dir_entry.config(state=state)
        self.new_lora_start_button.config(state=state)

        # Continue Training Existing LoRA controls
        self.existing_lora_combobox.config(state='readonly' if busy else 'normal')
        self.existing_lora_refresh_button.config(state=state)
        self.continue_lora_data_path_label.config(state=state)
        self.continue_lora_browse_button.config(state=state)
        self.continue_lora_output_dir_entry.config(state=state)
        self.continue_lora_start_button.config(state=state)

        # Merge and Import controls
        self.merge_ollama_combobox.config(state='readonly' if busy else 'normal')
        self.merge_ollama_refresh_button.config(state=state)
        self.merge_button.config(state=state)

    def is_busy(self):
        if (self.training_thread and self.training_thread.is_alive()) or \
           (self.merge_thread and self.merge_thread.is_alive()):
            messagebox.showwarning("警告", "一个任务正在进行中，请稍后再试。")
            return True
        return False

    def clear_logs(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def append_log(self, entry):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, entry + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()