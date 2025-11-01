import json
import time
import openai
from prompt_template import PromptTemplate


class MinuteWriter:
    def __init__(self, api_key, model_type):
        self.model_type = model_type
        if 'chatgpt' in model_type.lower():
            openai.api_key = api_key
            self.model = "gpt-3.5-turbo-0125"
        elif 'gpt4' in model_type.lower() or 'gpt-4' in model_type.lower():  # for gen & input corruption
            openai.api_key = api_key
            self.model = "gpt-4.1-mini-2025-04-14"
        else:  # llama, mistral, gemma, qwen (hf)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            cache_dir = "/data/jin62304/projects/hf_cache"  # Please set the huggingface cache directory ...

            model_type = model_type.replace("-hf", "")

            if 'qwen2.5' in model_type.lower():
                # 1.5B, 3B, 7B, 32B, 72B
                model_name = f"Qwen/{model_type}"
            elif 'mistral' in model_type.lower():
                model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                model_name = model_type

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=cache_dir
            )

    def hf_write(self, messages):
        print()
        print("=============================")
        print("Message input for API request: ")
        print(messages)
        print()

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.75,
            top_p=0.9,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return [response]

    def write(self, messages):
        print()
        print("=============================")
        print("# Message input for API request: ")
        print(messages)
        print()

        results = []
        retries = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=0.75,
                    top_p=0.9,
                    messages=messages
                )
                result = response['choices'][0]['message']['content']
                results.append(result)

                return results
            except:
                retries += 1
                print(f"Request failed. Retrying ({retries} …")
                time.sleep(2 * retries)
