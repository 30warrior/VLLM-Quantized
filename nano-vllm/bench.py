import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=None, help="Max sequence length (reduce for large models to save memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization target (e.g. 0.6 for 60%%)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Max tokens per batch")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Max concurrent sequences")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (needed for large models with INT8 linear layers)")
    parser.add_argument("--use-int8-linear", action="store_true", help="Replace all linear layers with bnb Linear8bitLt (for INT8 speedup on large models)")
    parser.add_argument("--use-int4-linear", action="store_true", help="Replace all linear layers with bnb Linear4bit (truly saves memory)")
    parser.add_argument("--use-custom-int4", action="store_true", help="Use custom offline INT4 quantization (saves ~75% weight memory)")
    parser.add_argument("--use-custom-int8", action="store_true", help="Use custom offline INT8 quantization (saves ~50% weight memory)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true", default=True)
    args = parser.parse_args()

    seed(0)
    num_seqs = args.num_requests
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser(args.model)
    llm_kwargs = dict(
        enforce_eager=args.enforce_eager,
        use_int8_linear=args.use_int8_linear,
        use_int4_linear=args.use_int4_linear,
        use_custom_int4=args.use_custom_int4,
        use_custom_int8=args.use_custom_int8,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(path, **llm_kwargs)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
