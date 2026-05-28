import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer #分词器


def main():
    # path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    path = os.path.expanduser("~/huggingface/Qwen2.5-7B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    # INT4量化加载（推荐，显存占用最低）
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        load_in_4bit=True,  # 开启INT4量化
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # INT8量化加载（取消注释以使用）
    # llm = LLM(path, enforce_eager=True, load_in_8bit=True)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 提示词转化为聊天模板格式
    outputs = llm.generate(prompts, sampling_params)
    # 返回一个字典列表，每个字典包含：
    #   - 'text': 生成的文本字符串
    #   - 'token_ids': 生成的token ID列表

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()