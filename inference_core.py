import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time

def load_model_and_tokenizer(adapter_path, status_queue):
    """
    加载基础模型、分词器，并应用LoRA适配器。
    通过队列报告加载状态。
    """
    try:
        # 1. 从适配器配置中获取基础模型名称
        config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = config.base_model_name_or_path
        status_queue.put(f"检测到基础模型: {base_model_name}")

        # 2. 加载分词器
        status_queue.put("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
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
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        status_queue.put("基础模型加载成功。")

        # 5. 加载并应用LoRA适配器
        status_queue.put("正在应用LoRA适配器...")
        model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
        model_with_lora.eval() # 设置为评估模式
        status_queue.put("模型准备就绪！")

        return model_with_lora, tokenizer

    except Exception as e:
        status_queue.put(f"错误: {e}")
        return None, None

def generate_response(model, tokenizer, instruction, input_text, history, temperature=0.8):
    """
    使用加载好的模型和分词器生成响应。
    新增了 history 参数用于处理多轮对话。
    """
    # 1. 构建符合Qwen-Chat模板的消息列表
    # system message 代表模型的角色指令
    messages = [{"role": "system", "content": instruction}]
    
    # 添加历史对话
    for user_turn, assistant_turn in history:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    
    # 添加当前的用户输入
    messages.append({"role": "user", "content": input_text})

    # 2. 使用 apply_chat_template 标准方法处理输入
    # 这是处理聊天模型输入的最佳实践，它会自动处理所有特殊的token和格式
    device = model.device
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # 确保在末尾添加 'assistant' 的提示，让模型知道该它说话了
        return_tensors="pt"
    ).to(device)

    # 3. 生成回答
    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_inputs,
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature
        )
    
    # 4. 解码新生成的部分
    # 为了得到干净的输出，我们只解码模型新生成的token，而不是整个序列
    response_tokens = outputs[0][len(model_inputs[0]):]
    generated_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
             
    return generated_output