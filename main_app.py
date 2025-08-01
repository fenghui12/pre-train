import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog
import threading
import queue
import time
import os
import json
from train_core import start_training, get_local_lora_base_models, get_existing_lora_dirs
from merge_and_import import do_merge_and_import, convert_base_model_to_ollama
from inference_core import load_model_and_tokenizer, generate_response, start_gradio_interface

CONFIG_FILE = "config.json"

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title("LLM 微调与管理助手 v2.4")
        self.parent.geometry("850x750")

        self.config = self.load_config()
        self.interactive_widgets = []

        # --- Queues for threading ---
        self.progress_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # --- Internal State ---
        self.active_thread = None
        self.selected_data_file = tk.StringVar()
        self.inference_model = None
        self.inference_tokenizer = None
        self.chat_history = []

        # --- Main PanedWindow for resizable layout ---
        self.main_paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame for Tabs ---
        self.notebook = ttk.Notebook(self.main_paned_window)
        self.main_paned_window.add(self.notebook, weight=1) # Let the notebook take most of the space

        # --- Create Tabs ---
        self.tab_train = ttk.Frame(self.notebook, padding="10")
        self.tab_manage = ttk.Frame(self.notebook, padding="10")
        self.tab_inference = ttk.Frame(self.notebook, padding="10")
        self.tab_settings = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_train, text="训练 (Train)")
        self.notebook.add(self.tab_manage, text="模型管理 (Manage)")
        self.notebook.add(self.tab_inference, text="推理 (Inference)")
        self.notebook.add(self.tab_settings, text="设置 (Settings)")

        # --- Bottom Frame for Logs and Progress ---
        # Give this a smaller, fixed initial size
        self.bottom_frame = ttk.Frame(self.main_paned_window, height=220)
        self.main_paned_window.add(self.bottom_frame, weight=0)

        # --- Progress Bar and Status Label ---
        self.progress_frame = ttk.LabelFrame(self.bottom_frame, text="任务进度", padding="5")
        self.progress_frame.pack(fill=tk.X, expand=False, padx=10, pady=(5,0))

        self.status_label = ttk.Label(self.progress_frame, text="状态: 空闲")
        self.status_label.pack(fill=tk.X, expand=True, padx=5, pady=2)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=2)

        # --- Log Viewer ---
        self.log_frame = ttk.LabelFrame(self.bottom_frame, text="实时日志", padding="5")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, state='disabled', wrap=tk.WORD, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Populate Tabs ---
        self.create_train_tab_content()
        self.create_manage_tab_content()
        self.create_inference_tab_content()
        self.create_settings_tab_content()

        # --- Start periodic check for thread queues ---
        self.parent.after(100, self.periodic_check)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {"llama_cpp_path": ""}

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def add_interactive_widget(self, widget):
        self.interactive_widgets.append(widget)

    def create_train_tab_content(self):
        self.train_mode = tk.StringVar(value="new")
        train_frame = ttk.Frame(self.tab_train)
        train_frame.pack(fill=tk.X, expand=False)

        mode_frame = ttk.LabelFrame(train_frame, text="1. 选择训练模式", padding="10")
        mode_frame.pack(fill=tk.X, expand=False, pady=5)
        
        rb_new = ttk.Radiobutton(mode_frame, text="新 LoRA 训练", variable=self.train_mode, value="new", command=self.on_train_mode_change)
        rb_new.pack(side=tk.LEFT, padx=5)
        rb_continue = ttk.Radiobutton(mode_frame, text="继续训练现有 LoRA", variable=self.train_mode, value="continue", command=self.on_train_mode_change)
        rb_continue.pack(side=tk.LEFT, padx=20)
        self.add_interactive_widget(rb_new)
        self.add_interactive_widget(rb_continue)

        self.model_select_frame = ttk.LabelFrame(train_frame, text="2. 选择模型", padding="10")
        self.model_select_frame.pack(fill=tk.X, expand=False, pady=5)

        self.model_select_label = ttk.Label(self.model_select_frame, text="基座模型:")
        self.model_select_label.pack(side=tk.LEFT, padx=(0, 5))
        self.model_select_combobox = ttk.Combobox(self.model_select_frame, state="normal", width=60)
        self.model_select_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.model_refresh_button = ttk.Button(self.model_select_frame, text="刷新", command=self.refresh_model_lists)
        self.model_refresh_button.pack(side=tk.RIGHT, padx=5)
        self.add_interactive_widget(self.model_select_combobox)
        self.add_interactive_widget(self.model_refresh_button)

        data_output_frame = ttk.LabelFrame(train_frame, text="3. 数据与输出", padding="10")
        data_output_frame.pack(fill=tk.X, expand=False, pady=5)

        data_frame = ttk.Frame(data_output_frame)
        data_frame.pack(fill=tk.X, expand=False, pady=5)
        data_path_label_desc = ttk.Label(data_frame, text="训练数据:")
        data_path_label_desc.pack(side=tk.LEFT, padx=(0, 5))
        self.data_path_label = ttk.Label(data_frame, text="未选择文件", width=70, relief="sunken")
        self.data_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        browse_button = ttk.Button(data_frame, text="浏览...", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT, padx=5)
        self.add_interactive_widget(browse_button)

        output_frame = ttk.Frame(data_output_frame)
        output_frame.pack(fill=tk.X, expand=False, pady=5)
        output_dir_label = ttk.Label(output_frame, text="输出目录:")
        output_dir_label.pack(side=tk.LEFT, padx=(0, 5))
        self.output_dir_entry = ttk.Entry(output_frame, width=70)
        self.output_dir_entry.pack(fill=tk.X, expand=True, padx=5)
        self.add_interactive_widget(self.output_dir_entry)

        self.start_train_button = ttk.Button(train_frame, text="开始训练", command=self.start_training_thread, style="Accent.TButton")
        self.start_train_button.pack(pady=20, ipady=5)
        self.add_interactive_widget(self.start_train_button)

        self.on_train_mode_change()

    def create_manage_tab_content(self):
        manage_frame = ttk.Frame(self.tab_manage)
        manage_frame.pack(fill=tk.X, expand=False)

        merge_frame = ttk.LabelFrame(manage_frame, text="1. 合并 LoRA 并导入 Ollama", padding="10")
        merge_frame.pack(fill=tk.X, expand=False, pady=5)

        merge_model_frame = ttk.Frame(merge_frame)
        merge_model_frame.pack(fill=tk.X, expand=True, pady=5)
        merge_model_label = ttk.Label(merge_model_frame, text="选择要合并的LoRA模型:")
        merge_model_label.pack(side=tk.LEFT, padx=(0, 5))
        self.merge_model_combobox = ttk.Combobox(merge_model_frame, state="readonly", width=60)
        self.merge_model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.merge_refresh_button = ttk.Button(merge_model_frame, text="刷新", command=self.refresh_merge_model_list)
        self.merge_refresh_button.pack(side=tk.RIGHT, padx=5)
        self.add_interactive_widget(self.merge_model_combobox)
        self.add_interactive_widget(self.merge_refresh_button)

        merge_button = ttk.Button(merge_frame, text="开始合并与导入", command=self.start_merge_and_import_thread, style="Accent.TButton")
        merge_button.pack(pady=10)
        self.add_interactive_widget(merge_button)

        convert_frame = ttk.LabelFrame(manage_frame, text="2. 转换基座模型到 Ollama", padding="10")
        convert_frame.pack(fill=tk.X, expand=False, pady=5)
        
        convert_desc_label = ttk.Label(convert_frame, text="此功能可将任意 Hugging Face 模型转换为 Ollama 格式。", wraplength=600)
        convert_desc_label.pack(pady=(0, 10))

        convert_button = ttk.Button(convert_frame, text="开始转换", command=self.start_convert_base_model_thread, style="Accent.TButton")
        convert_button.pack(pady=10)
        self.add_interactive_widget(convert_button)

        self.refresh_merge_model_list()

    def create_inference_tab_content(self):
        inference_frame = ttk.Frame(self.tab_inference)
        inference_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(inference_frame)
        top_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

        model_frame = ttk.LabelFrame(top_frame, text="1. 选择并加载模型", padding="10")
        model_frame.pack(fill=tk.X, expand=True)

        self.inference_model_combobox = ttk.Combobox(model_frame, state="readonly", width=60)
        self.inference_model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.inference_refresh_button = ttk.Button(model_frame, text="刷新", command=self.refresh_inference_model_list)
        self.inference_refresh_button.pack(side=tk.LEFT, padx=5)
        self.load_inference_model_button = ttk.Button(model_frame, text="加载模型", command=self.load_inference_model_thread, style="Accent.TButton")
        self.load_inference_model_button.pack(side=tk.RIGHT, padx=5)
        self.add_interactive_widget(self.inference_model_combobox)
        self.add_interactive_widget(self.inference_refresh_button)
        self.add_interactive_widget(self.load_inference_model_button)

        # Add Share Model button
        self.share_model_button = ttk.Button(model_frame, text="分享模型 (Gradio)", command=self.start_gradio_share_thread, style="Accent.TButton", state=tk.DISABLED)
        self.share_model_button.pack(side=tk.RIGHT, padx=5)
        self.add_interactive_widget(self.share_model_button)

        system_prompt_frame = ttk.LabelFrame(top_frame, text="2. 系统指令 (System Prompt)", padding="10")
        system_prompt_frame.pack(fill=tk.X, expand=True, pady=(10, 0))
        self.system_prompt_entry = ttk.Entry(system_prompt_frame)
        self.system_prompt_entry.insert(0, "You are a helpful assistant.")
        self.system_prompt_entry.pack(fill=tk.X, expand=True)
        self.add_interactive_widget(self.system_prompt_entry)

        chat_frame = ttk.LabelFrame(inference_frame, text="3. 对话", padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_history_text = scrolledtext.ScrolledText(chat_frame, state='disabled', wrap=tk.WORD, height=15)
        self.chat_history_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_history_text.tag_config('user_style', foreground="#0078d4", font=("Segoe UI", 10, "bold"))
        # Use a theme-aware color for the model response
        style = ttk.Style()
        model_fg_color = style.lookup('TLabel', 'foreground')
        self.chat_history_text.tag_config('model_style', foreground=model_fg_color, font=("Segoe UI", 10, "bold"))

        action_frame = ttk.Frame(chat_frame)
        action_frame.pack(fill=tk.X, expand=False, pady=(5, 0))

        self.user_input_text = scrolledtext.ScrolledText(action_frame, height=4, wrap=tk.WORD)
        self.user_input_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 10))
        self.add_interactive_widget(self.user_input_text)

        controls_frame = ttk.Frame(action_frame)
        controls_frame.pack(side=tk.RIGHT, anchor='n')

        self.send_button = ttk.Button(controls_frame, text="发送", command=self.send_message_thread, style="Accent.TButton")
        self.send_button.pack(fill=tk.X, expand=True, ipady=5)
        self.add_interactive_widget(self.send_button)

        self.context_mode_var = tk.BooleanVar(value=True)
        context_check = ttk.Checkbutton(controls_frame, text="上下文", variable=self.context_mode_var)
        context_check.pack(pady=5, anchor='w')
        self.add_interactive_widget(context_check)

        clear_history_button = ttk.Button(controls_frame, text="清空", command=self.clear_chat_history)
        clear_history_button.pack(fill=tk.X, expand=True)
        self.add_interactive_widget(clear_history_button)

        self.refresh_inference_model_list()

    def create_settings_tab_content(self):
        settings_frame = ttk.Frame(self.tab_settings)
        settings_frame.pack(fill=tk.X, expand=False, pady=5)

        llama_cpp_frame = ttk.LabelFrame(settings_frame, text="路径配置", padding="10")
        llama_cpp_frame.pack(fill=tk.X, expand=True)

        llama_cpp_label = ttk.Label(llama_cpp_frame, text="llama.cpp 仓库路径:")
        llama_cpp_label.pack(side=tk.LEFT, padx=(0, 5))
        self.llama_cpp_path_entry = ttk.Entry(llama_cpp_frame, width=70)
        self.llama_cpp_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.llama_cpp_path_entry.insert(0, self.config.get("llama_cpp_path", ""))
        browse_llama_button = ttk.Button(llama_cpp_frame, text="浏览...", command=self.browse_llama_cpp_path)
        browse_llama_button.pack(side=tk.LEFT, padx=5)
        self.add_interactive_widget(self.llama_cpp_path_entry)
        self.add_interactive_widget(browse_llama_button)

        save_button = ttk.Button(settings_frame, text="保存设置", command=self.save_settings, style="Accent.TButton")
        save_button.pack(pady=20)
        self.add_interactive_widget(save_button)

    def set_ui_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        for widget in self.interactive_widgets:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass
        # Special handling for share_model_button
        if hasattr(self, 'share_model_button'):
            if busy or self.inference_model is None:
                self.share_model_button.config(state=tk.DISABLED)
            else:
                self.share_model_button.config(state=tk.NORMAL)

    def update_chat_display(self, role, text):
        self.chat_history_text.config(state='normal')
        style_tag = f'{role}_style'
        self.chat_history_text.insert(tk.END, f"{role.capitalize()}:\n", style_tag)
        self.chat_history_text.insert(tk.END, f"{text}\n\n")
        self.chat_history_text.config(state='disabled')
        self.chat_history_text.yview(tk.END)

    def on_train_mode_change(self):
        mode = self.train_mode.get()
        if mode == "new":
            self.model_select_label.config(text="基座模型:")
            self.model_select_combobox.set("请选择或输入 Hugging Face 模型 ID...")
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, "./lora_finetuned_model")
        else: # continue
            self.model_select_label.config(text="选择现有 LoRA:")
            self.model_select_combobox.set("扫描本地已训练 LoRA 模型...")
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, "./lora_continued_model")
        self.refresh_model_lists()

    def refresh_model_lists(self):
        self.model_refresh_button.config(state=tk.DISABLED)
        mode = self.train_mode.get()
        if mode == "new":
            self.model_select_combobox.set("正在扫描本地基座模型...")
            models, error = get_local_lora_base_models()
            if error:
                messagebox.showerror("错误", error)
                self.model_select_combobox.set("扫描失败")
                self.model_select_combobox['values'] = []
            else:
                self.model_select_combobox['values'] = models
                if models:
                    self.model_select_combobox.set(models[0])
                else:
                    self.model_select_combobox.set("未找到本地缓存的基座模型")
        else: # continue
            self.model_select_combobox.set("正在扫描本地已训练 LoRA 模型...")
            models, error = get_existing_lora_dirs()
            if error:
                messagebox.showerror("错误", error)
                self.model_select_combobox.set("扫描失败")
                self.model_select_combobox['values'] = []
            elif not models:
                self.model_select_combobox.set("未找到本地已训练 LoRA 模型")
                self.model_select_combobox['values'] = []
            else:
                self.model_select_combobox['values'] = models
                self.model_select_combobox.set(models[0])
        self.model_refresh_button.config(state=tk.NORMAL)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择数据集文件",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*" ))
        )
        if file_path:
            self.selected_data_file.set(file_path)
            self.data_path_label.config(text=file_path)
            filename = os.path.basename(file_path).split('.')[0]
            mode = self.train_mode.get()
            suffix = "finetuned" if mode == "new" else "continued"
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, f"./lora_{filename}_{suffix}")

    def start_training_thread(self):
        if self.is_busy("训练"): return

        mode = self.train_mode.get()
        model_path = self.model_select_combobox.get().strip()
        data_path = self.selected_data_file.get()
        output_dir = self.output_dir_entry.get().strip()

        if not model_path or "扫描失败" in model_path or "未找到" in model_path:
            messagebox.showerror("错误", "请先选择一个有效的模型或LoRA路径！")
            return
        if not data_path:
            messagebox.showerror("错误", "请先选择一个数据集文件！")
            return
        if not output_dir:
            messagebox.showerror("错误", "请输入一个输出目录名！")
            return

        base_model_name = None
        lora_adapter_path = None

        if mode == 'new':
            base_model_name = model_path
        else: # continue
            lora_adapter_path = os.path.join(model_path, "final_lora_adapter")
            if not os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
                messagebox.showerror("错误", f"在 '{model_path}' 中未找到有效的 'final_lora_adapter' 子目录或其配置文件。")
                return
            try:
                with open(os.path.join(lora_adapter_path, "adapter_config.json"), 'r') as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                if not base_model_name:
                    raise ValueError("无法获取基座模型名称")
            except Exception as e:
                messagebox.showerror("错误", f"读取LoRA适配器配置失败: {e}")
                return

        self.set_ui_busy(True)
        self.clear_logs()
        self.status_label.config(text=f"状态: 准备开始训练...")

        self.active_thread = threading.Thread(
            target=start_training,
            args=(base_model_name, data_path, output_dir, self.progress_queue, self.log_queue, lora_adapter_path),
            daemon=True
        )
        self.active_thread.start()

    def refresh_merge_model_list(self):
        self.merge_refresh_button.config(state=tk.DISABLED)
        self.merge_model_combobox.set("正在扫描本地已训练 LoRA 模型...")
        models, error = get_existing_lora_dirs()
        if error:
            messagebox.showerror("错误", error)
            self.merge_model_combobox.set("扫描失败")
            self.merge_model_combobox['values'] = []
        elif not models:
            self.merge_model_combobox.set("未找到本地已训练 LoRA 模型")
            self.merge_model_combobox['values'] = []
        else:
            self.merge_model_combobox['values'] = models
            self.merge_model_combobox.set(models[0])
        self.merge_refresh_button.config(state=tk.NORMAL)

    def start_merge_and_import_thread(self):
        if self.is_busy("合并"): return

        adapter_dir = self.merge_model_combobox.get().strip()
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

        self.active_thread = threading.Thread(
            target=do_merge_and_import,
            args=(final_adapter_path, ollama_model_name, self.status_queue.put),
            daemon=True
        )
        self.active_thread.start()

    def start_convert_base_model_thread(self):
        if self.is_busy("转换"): return

        current_model = self.model_select_combobox.get().strip()
        if "扫描失败" in current_model or "未找到" in current_model:
            current_model = ""

        base_model_id = simpledialog.askstring(
            "输入 Hugging Face 模型 ID", 
            "请输入要转换的 Hugging Face 模型 ID (如果模型不在本地，将自动下载):",
            initialvalue=current_model
        )
        
        if not base_model_id:
            return

        base_model_id = base_model_id.strip()
        default_name = base_model_id.split('/')[-1].lower() + ":latest"
        ollama_model_name = simpledialog.askstring("输入 Ollama 模型名称", "请输入您想在 Ollama 中使用的新模型名称:", initialvalue=default_name)
        if not ollama_model_name:
            return

        self.set_ui_busy(True)
        self.clear_logs()
        self.status_label.config(text=f"状态: 正在转换基座模型 {base_model_id}...")

        self.active_thread = threading.Thread(
            target=convert_base_model_to_ollama,
            args=(base_model_id, ollama_model_name, self.status_queue.put),
            daemon=True
        )
        self.active_thread.start()

    def refresh_inference_model_list(self):
        self.inference_refresh_button.config(state=tk.DISABLED)
        self.inference_model_combobox.set("正在扫描所有可用模型...")
        
        base_models, error1 = get_local_lora_base_models()
        lora_models, error2 = get_existing_lora_dirs()
        
        all_models = []
        if not error1:
            all_models.extend(base_models)
        if not error2:
            all_models.extend(lora_models)

        if not all_models:
            self.inference_model_combobox.set("未找到任何可用模型")
            self.inference_model_combobox['values'] = []
        else:
            self.inference_model_combobox['values'] = sorted(list(set(all_models)))
            self.inference_model_combobox.set(all_models[0])
        self.inference_refresh_button.config(state=tk.NORMAL)

    def load_inference_model_thread(self):
        if self.is_busy("加载模型"): return

        model_path = self.inference_model_combobox.get().strip()
        if not model_path or "未找到" in model_path:
            messagebox.showerror("错误", "请选择一个有效的模型进行加载！")
            return

        self.set_ui_busy(True)
        self.clear_logs()
        self.status_label.config(text=f"状态: 正在加载模型 {os.path.basename(model_path)}...")
        self.progress_bar.start()

        self.active_thread = threading.Thread(
            target=self.load_inference_model,
            args=(model_path,),
            daemon=True
        )
        self.active_thread.start()

    def load_inference_model(self, model_path):
        adapter_path = os.path.join(model_path, "final_lora_adapter")
        if not os.path.exists(adapter_path):
            adapter_path = model_path

        self.inference_model, self.inference_tokenizer = load_model_and_tokenizer(adapter_path, self.status_queue)
        if self.inference_model is None:
            self.status_queue.put("ERROR: 模型加载失败，请检查日志。")
            self.share_model_button.config(state=tk.DISABLED) # Disable share button on failure
        else:
            self.status_queue.put(f"SUCCESS: 模型 {os.path.basename(model_path)} 加载成功！")
            self.share_model_button.config(state=tk.NORMAL) # Enable share button on success

    def start_gradio_share_thread(self):
        if self.is_busy("分享模型"): return
        if self.inference_model is None or self.inference_tokenizer is None:
            messagebox.showerror("错误", "请先成功加载一个模型才能分享！")
            return

        system_prompt = self.system_prompt_entry.get().strip()

        self.set_ui_busy(True)
        self.clear_logs()
        self.status_label.config(text="状态: 正在启动 Gradio 服务并分享模型...")
        self.progress_bar.start()

        # Gradio 启动后会阻塞，所以必须在新线程中运行
        self.active_thread = threading.Thread(
            target=start_gradio_interface,
            args=(self.inference_model, self.inference_tokenizer, system_prompt, self.status_queue.put),
            daemon=True
        )
        self.active_thread.start()

    def send_message_thread(self):
        if self.is_busy("生成回复"): return
        if self.inference_model is None:
            messagebox.showerror("错误", "请先成功加载一个模型！")
            return

        user_message = self.user_input_text.get("1.0", tk.END).strip()
        if not user_message:
            messagebox.showwarning("警告", "请输入内容！")
            return

        self.update_chat_display("user", user_message)
        self.user_input_text.delete("1.0", tk.END)

        self.set_ui_busy(True)
        self.status_label.config(text="状态: 模型正在思考中...")
        self.progress_bar.start()

        history_to_send = self.chat_history if self.context_mode_var.get() else []
        system_prompt = self.system_prompt_entry.get().strip()

        self.active_thread = threading.Thread(
            target=self.run_generation,
            args=(system_prompt, user_message, history_to_send),
            daemon=True
        )
        self.active_thread.start()

    def run_generation(self, instruction, input_text, history):
        response = generate_response(self.inference_model, self.inference_tokenizer, instruction, input_text, history)
        self.response_queue.put((input_text, response))

    def clear_chat_history(self):
        self.chat_history = []
        self.chat_history_text.config(state='normal')
        self.chat_history_text.delete(1.0, tk.END)
        self.chat_history_text.config(state='disabled')
        messagebox.showinfo("操作成功", "对话历史已清空。")

    def browse_llama_cpp_path(self):
        dir_path = filedialog.askdirectory(title="选择 llama.cpp 仓库的根目录")
        if dir_path:
            self.llama_cpp_path_entry.delete(0, tk.END)
            self.llama_cpp_path_entry.insert(0, dir_path)

    def save_settings(self):
        self.config["llama_cpp_path"] = self.llama_cpp_path_entry.get().strip()
        self.save_config()
        messagebox.showinfo("成功", "设置已保存！")

    def periodic_check(self):
        # Check training progress queue
        while not self.progress_queue.empty():
            data = self.progress_queue.get_nowait()
            if 'error' in data:
                messagebox.showerror("训练失败", f"发生错误: {data['error']}")
                self.set_ui_busy(False)
                return

            self.progress_bar.stop()
            self.progress_bar['value'] = data['progress']
            loss = data.get('loss', 'N/A')
            eta_seconds = data.get('eta_seconds', float('inf'))
            eta_str = "计算中..." if eta_seconds == float('inf') else time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            status_text = f"进度: {data['progress']:.2f}% | 当前 Loss: {loss} | 预计剩余时间: {eta_str}"
            self.status_label.config(text=status_text)
            
            if data.get('done'):
                messagebox.showinfo("成功", "训练已成功完成！")
                self.set_ui_busy(False)

        # Check general status queue (for merge, convert, model loading)
        while not self.status_queue.empty():
            log_entry = self.status_queue.get_nowait()
            # 注意：不要在这里调用self.append_log，因为在某些条件下会重复添加
            self.status_label.config(text=f"状态: {log_entry}")
            
            # 检查是否有错误消息
            if "ERROR:" in log_entry or "Connection aborted" in log_entry or "ConnectionError" in log_entry:
                self.append_log(log_entry)
                messagebox.showerror("失败", log_entry)
                self.set_ui_busy(False)
                self.progress_bar.stop()
                self.status_label.config(text="状态: 空闲")
            # 检查所有可能的成功消息
            elif "SUCCESS:" in log_entry or "模型准备就绪" in log_entry:
                # 这是最终的成功状态，释放UI
                self.append_log(log_entry)
                messagebox.showinfo("成功", log_entry)
                self.set_ui_busy(False)
                self.progress_bar.stop()
                self.status_label.config(text="状态: 空闲")
            elif "分词器加载成功" in log_entry or "基础模型加载成功" in log_entry or "LoRA适配器应用成功" in log_entry:
                # 这些是加载过程中的中间状态，添加到日志但不弹出消息框
                self.append_log(log_entry)
            elif "Gradio 界面已启动，公网分享链接:" in log_entry or "Gradio 界面已成功分享！" in log_entry:
                # 特殊处理Gradio分享相关消息
                self.append_log(log_entry)
                messagebox.showinfo("成功", log_entry)
                self.set_ui_busy(False)
                self.progress_bar.stop()
                self.status_label.config(text="状态: 空闲")
            else:
                # 其他消息也添加到日志中
                self.append_log(log_entry)

        # Check inference response queue
        while not self.response_queue.empty():
            user_message, model_response = self.response_queue.get_nowait()
            self.update_chat_display("model", model_response)
            if self.context_mode_var.get():
                self.chat_history.append((user_message, model_response))
            self.set_ui_busy(False)
            self.progress_bar.stop()
            self.status_label.config(text="状态: 空闲")

        # Check log queue
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get_nowait())

        self.parent.after(100, self.periodic_check)

    def is_busy(self, task_name="任务"):
        is_alive = self.active_thread and self.active_thread.is_alive()
        if is_alive:
            messagebox.showwarning("警告", f"一个{task_name}正在进行中，请稍后再试。")
        return is_alive

    def clear_logs(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def append_log(self, entry):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, str(entry) + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("source", "azure.tcl")
        style = ttk.Style(root)
        style.theme_use("azure-dark")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    except tk.TclError:
        print("Azure theme not found, using default.")

    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()