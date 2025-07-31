import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time
import os

def load_model_and_tokenizer(model_path, status_queue):
    """
    加载模型和分词器。
    如果 model_path 指向一个 LoRA 适配器目录，则加载基础模型并应用适配器。
    如果 model_path 是一个 Hugging Face 模型ID，则直接加载该基座模型。
    通过队列报告加载状态。
    """
    try:
        is_lora_adapter = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
        base_model_name = model_path
        
        if is_lora_adapter:
            status_queue.put(f"检测到 LoRA 适配器: {model_path}")
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            status_queue.put(f"基础模型为: {base_model_name}")
        else:
            status_queue.put(f"准备直接加载基座模型: {model_path}")

        # 2. 加载分词器
        status_queue.put("正在加载分词器...")
        tokenizer_path = model_path if is_lora_adapter else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        status_queue.put("分词器加载成功。")

        # 3. 配置量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 4. 加载基础模型
        status_queue.put("正在加载量化的基础模型 (可能需要几分钟)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        status_queue.put("基础模型加载成功。")

        # 5. 如果是LoRA，则应用适配器
        if is_lora_adapter:
            status_queue.put("正在应用LoRA适配器...")
            model = PeftModel.from_pretrained(model, model_path)
            status_queue.put("LoRA适配器应用成功。")
        
        model.eval() # 设置为评估模式
        status_queue.put("模型准备就绪！")

        return model, tokenizer

    except Exception as e:
        status_queue.put(f"错误: {e}")
        import traceback
        status_queue.put(traceback.format_exc())
        return None, None

def generate_response(model, tokenizer, instruction, input_text, history, temperature=0.8):
    """
    使用加载好的模型和分词器生成响应。
    """
    messages = [{"role": "system", "content": instruction}]
    
    for user_turn, assistant_turn in history:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    
    messages.append({"role": "user", "content": input_text})

    device = model.device
    # apply_chat_template 会返回一个包含 input_ids 和 attention_mask 的字典
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # 明确传递 attention_mask 以避免警告
        outputs = model.generate(
            input_ids=model_inputs,
            attention_mask=torch.ones_like(model_inputs), # 明确传递 attention_mask
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature
        )
    
    response_tokens = outputs[0][len(model_inputs[0]):]
    generated_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
             
    return generated_output