import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time
import os
import gradio as gr
import requests
from requests.exceptions import ConnectionError, Timeout


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
            # 使用本地文件读取而不是 from_pretrained 来避免网络请求
            import json
            config_path = os.path.join(model_path, 'adapter_config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            base_model_name = config_data.get("base_model_name_or_path", model_path)
            status_queue.put(f"基础模型为: {base_model_name}")
        else:
            status_queue.put(f"准备直接加载基座模型: {model_path}")

        # 2. 加载分词器
        status_queue.put("正在加载分词器...")
        tokenizer_path = model_path if is_lora_adapter else base_model_name
        
        # 添加重试机制
        tokenizer = None
        for attempt in range(3):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
                break
            except (ConnectionError, Timeout) as e:
                status_queue.put(f"网络连接错误 (尝试 {attempt + 1}/3): {str(e)}")
                if attempt < 2:  # 不是最后一次尝试
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise  # 最后一次尝试失败，重新抛出异常
            except Exception as e:
                # 其他非网络错误直接抛出
                raise
        
        if tokenizer is None:
            raise Exception("无法加载分词器")
            
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
        model = None
        for attempt in range(3):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                    local_files_only=True
                )
                break
            except (ConnectionError, Timeout) as e:
                status_queue.put(f"网络连接错误 (尝试 {attempt + 1}/3): {str(e)}")
                if attempt < 2:  # 不是最后一次尝试
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise  # 最后一次尝试失败，重新抛出异常
            except Exception as e:
                # 其他非网络错误直接抛出
                raise
                
        if model is None:
            raise Exception("无法加载基础模型")
            
        status_queue.put("基础模型加载成功。")

        # 5. 如果是LoRA，则应用适配器
        if is_lora_adapter:
            status_queue.put("正在应用LoRA适配器...")
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
            status_queue.put("LoRA适配器应用成功。")
        
        model.eval()  # 设置为评估模式
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

def start_gradio_interface(model, tokenizer, system_prompt, status_callback):
    """
    使用 Gradio 启动一个聊天界面，并分享到公网。
    """
    status_callback("正在启动 Gradio 界面...")

    def gradio_chat_wrapper(message, history):
        # Gradio history format when type="messages":
        # [{"role": "user", "content": "user_msg1"}, {"role": "assistant", "content": "bot_msg1"}, ...]
        # Our generate_response history format: [(user_msg1, bot_msg1), (user_msg2, bot_msg2), ...]

        converted_history = []
        # Iterate through history in pairs (user, assistant)
        # history is a list of dictionaries, where each dictionary is a message
        # We need to extract user and assistant messages in pairs
        for i in range(0, len(history), 2):
            user_msg_dict = history[i]
            assistant_msg_dict = history[i+1]
            converted_history.append((user_msg_dict["content"], assistant_msg_dict["content"]))

        response = generate_response(model, tokenizer, system_prompt, message, converted_history)
        
        return response

    try:
        from threading import Thread
        
        def launch_and_share():
            chat_interface = gr.ChatInterface(
                fn=gradio_chat_wrapper,
                title="LLM 微调模型分享",
                description=f"当前模型: {model.config._name_or_path.split('/')[-1]}",
                examples=[["你好，请介绍一下你自己。", []]],
                theme="soft",
                chatbot=gr.Chatbot(height=400, type="messages"),
                textbox=gr.Textbox(placeholder="输入你的消息...", container=False, scale=7)
            )
            
            # 启动 Gradio 界面并分享
            status_callback("正在启动 Gradio 服务...")
            share_url = chat_interface.launch(share=True, quiet=True, prevent_thread_lock=True)
            status_callback(f"Gradio 界面已启动，公网分享链接: {share_url}")
            status_callback("SUCCESS: Gradio 界面已成功分享！")
            
            # 保持服务运行
            try:
                from threading import Event
                Event().wait()  # 阻止线程退出，保持服务运行
            except KeyboardInterrupt:
                pass
                
        # 在新线程中启动Gradio服务
        gradio_thread = Thread(target=launch_and_share, daemon=True)
        gradio_thread.start()
        
    except Exception as e:
        status_callback(f"ERROR: 启动 Gradio 界面失败: {e}")
        import traceback
        status_callback(traceback.format_exc())
