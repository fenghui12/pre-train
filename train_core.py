import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from transformers.trainer_callback import TrainerCallback
import os
import logging
import json
from peft import PeftConfig # 导入 PeftConfig

# --- Local LoRA Model Integration ---
def get_local_lora_base_models(base_path="."):
    """
    扫描指定路径下所有包含 'final_lora_adapter' 子目录的路径，
    并从其 adapter_config.json 中提取原始基座模型的名称。
    返回一个包含这些唯一基座模型名称的列表。
    """
    base_model_ids = set() # 使用集合来存储唯一的模型ID
    for root, dirs, files in os.walk(base_path):
        if "final_lora_adapter" in dirs:
            lora_adapter_dir = os.path.join(root, "final_lora_adapter")
            adapter_config_path = os.path.join(lora_adapter_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, 'r') as f:
                        config = json.load(f)
                        base_model_name = config.get("base_model_name_or_path")
                        if base_model_name:
                            base_model_ids.add(base_model_name)
                except Exception as e:
                    logging.warning(f"无法读取或解析 {adapter_config_path}: {e}")
    return sorted(list(base_model_ids)), None # 返回排序后的列表和None

def get_existing_lora_dirs(base_path="."):
    """
    扫描指定路径下所有包含 'final_lora_adapter' 子目录的父目录路径。
    返回一个包含这些 LoRA 模型目录路径的列表。
    """
    existing_lora_dirs = []
    for root, dirs, files in os.walk(base_path):
        if "final_lora_adapter" in dirs:
            lora_model_path = os.path.abspath(root)
            # 再次检查 adapter_config.json 确保是有效的 LoRA 模型目录
            if os.path.exists(os.path.join(lora_model_path, "final_lora_adapter", "adapter_config.json")):
                existing_lora_dirs.append(lora_model_path)
    return sorted(existing_lora_dirs), None # 返回排序后的列表和None

# 定义一个自定义的回调类，用于将进度更新传给GUI
class ProgressCallback(TrainerCallback):
    def __init__(self, progress_queue):
        self.progress_queue = progress_queue
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        # 计算进度百分比
        progress = (state.global_step / state.max_steps) * 100
        # 计算预估剩余时间
        elapsed_time = time.time() - self.start_time
        if state.global_step > 0:
            time_per_step = elapsed_time / state.global_step
            remaining_steps = state.max_steps - state.global_step
            eta = time_per_step * remaining_steps
        else:
            eta = float('inf')
        
        # 将进度和ETA放入队列
        self.progress_queue.put({'progress': progress, 'eta_seconds': eta, 'loss': state.log_history[-1]['loss'] if state.log_history else 'N/A'})

# 主训练函数，接收GUI传来的参数和回调
def start_training(base_model_name, data_path, output_dir, progress_queue, log_queue, lora_adapter_path=None):
    
    # --- 日志重定向 ---
    # 创建一个处理器，将日志消息发送到队列
    class QueueHandler(logging.Handler):
        def __init__(self, log_queue):
            super().__init__()
            self.log_queue = log_queue
        
        def emit(self, record):
            self.log_queue.put(self.format(record))

    # 配置根日志记录器
    # 使用 getLogger() 获取或创建记录器，避免重复添加处理器
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    # 清理旧的处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 添加我们的队列处理器和文件处理器
    logger.addHandler(QueueHandler(log_queue))
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "training_log.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    try:
        logger.info("训练流程开始。")
        logger.info(f"使用基础模型: {base_model_name}")
        if lora_adapter_path:
            logger.info(f"使用 LoRA 适配器: {lora_adapter_path}")
        logger.info(f"数据路径: {data_path}")
        logger.info(f"输出目录: {output_dir}")

        # 1. 加载数据集
        logger.info("步骤 1: 加载并处理数据集...")
        raw_dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info(f"成功加载 {len(raw_dataset)} 条数据。")

        # 2. 加载分词器
        logger.info(f"步骤 2: 加载分词器 ({base_model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer's pad_token 设置为 eos_token: {tokenizer.eos_token}")

        # 3. 定义处理函数
        def process_func(example):
            # 构建符合Qwen-Chat模板的消息列表
            messages = [
                {"role": "system", "content": example['instruction']},
                {"role": "user", "content": example['input']},
                {"role": "assistant", "content": example['output']}
            ]
            
            # 对完整的对话进行分词
            tokenized_full = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False, # 训练时不需要额外的生成提示
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            # 确保 tokenized_full 是一个 BatchEncoding 对象，并且有 input_ids 属性
            # 或者是一个非空的 Tensor
            labels = None
            if isinstance(tokenized_full, dict) and 'input_ids' in tokenized_full:
                labels = tokenized_full['input_ids'].clone()
            elif isinstance(tokenized_full, torch.Tensor) and tokenized_full.numel() > 0:
                labels = tokenized_full.clone()
            else:
                raise ValueError("Unexpected type or empty output for tokenized_full from tokenizer.apply_chat_template")
            
            # 对提示部分（system + user）进行分词，用于计算标签的忽略位置
            prompt_messages = [
                {"role": "system", "content": example['instruction']},
                {"role": "user", "content": example['input']}
            ]
            
            # 使用 apply_chat_template 处理提示部分
            tokenized_prompt_output = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True, # 确保这里包含 assistant 提示，与推理时一致
                truncation=True,
                max_length=1024, # 确保截断，避免过长序列
                return_tensors="pt"
            )
            
            # 确保 tokenized_prompt_output 是一个 BatchEncoding 对象，并且有 input_ids 属性
            # 或者是一个非空的 Tensor
            prompt_len = 0
            if isinstance(tokenized_prompt_output, dict) and 'input_ids' in tokenized_prompt_output:
                # 如果是 BatchEncoding 字典形式
                prompt_len = tokenized_prompt_output['input_ids'].shape[1]
            elif isinstance(tokenized_prompt_output, torch.Tensor) and tokenized_prompt_output.numel() > 0:
                # 如果是 Tensor 形式且非空
                prompt_len = tokenized_prompt_output.shape[1]
            
            # 创建标签，将提示部分的标签设为-100
            # 确保 prompt_len 不会超出 labels 的维度
            if prompt_len > 0 and labels is not None and prompt_len <= labels.shape[1]:
                labels[:, :prompt_len] = -100
            
            return {
                "input_ids": tokenized_full.input_ids.squeeze(0) if isinstance(tokenized_full, dict) else tokenized_full.squeeze(0), # 移除批次维度
                "labels": labels.squeeze(0) # 移除批次维度
            }

        # 5. 处理数据集
        tokenized_dataset = raw_dataset.map(process_func, remove_columns=raw_dataset.column_names)
        logger.info("数据集处理完毕。")

        # 6. 配置4-bit量化
        logger.info("步骤 3: 配置4-bit量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 7. 加载基础模型
        logger.info(f"步骤 4: 加载基础模型 ({base_model_name}) 并量化...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto" # 自动选择设备
        )
        model = prepare_model_for_kbit_training(model)
        logger.info("基础模型加载并量化完毕。")

        # 8. 配置 LoRA
        logger.info("步骤 5: 配置 LoRA...")
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            logger.info(f"加载并应用现有 LoRA 适配器: {lora_adapter_path}")
            from peft import PeftModel # 确保导入 PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter_path)
        else:
            logger.info("从头开始创建新的 LoRA 适配器。")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 9. 配置训练参数
        logger.info("步骤 6: 配置训练参数...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=1,
            num_train_epochs=8,
            learning_rate=2e-4,
            save_strategy="epoch",
            save_total_limit=2,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True, 
            tf32=True, 
        )

        # 10. 创建 Trainer
        logger.info("步骤 7: 创建 Trainer 并开始训练...")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(progress_queue)]
        )
        
        # 11. 开始训练
        trainer.train()

        # 12. 保存最终的适配器
        final_adapter_dir = os.path.join(output_dir, "final_lora_adapter")
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info(f"训练完成！最终适配器已保存至: {final_adapter_dir}")
        progress_queue.put({'progress': 100, 'eta_seconds': 0, 'loss': 'N/A', 'done': True})
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        progress_queue.put({'progress': -1, 'error': str(e)})
