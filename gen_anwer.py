from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import json
import torch
import argparse
from tqdm import tqdm

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--prompt_path", type=str, default="prompt/")
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--dataset_path", type=str, default="data/")
    parser.add_argument("--benchmark_type", type=str)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--num_choices", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    
    return parser.parse_args()

def load_benchmark(dataset_path, benchmark_type):
    questions = []
    with open(dataset_path + benchmark_type + ".json", "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def load_template(prompt_path, prompt_type):
    with open(prompt_path + prompt_type + "_prompt.json", "r") as tem_file:
        template = json.load(tem_file)
        return template

def get_model_outputs(args, questions, template):
    llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype=args.dtype
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

    # 지금은 1st turn 하나 생성하고 바로 이어서 2nd turn 생성
    prompts = []
    generations = []

    for i in range(len(questions[0]["content"])):
        prompt = []
        for j, qs in enumerate(questions):
            if i is 0:
                p = template["system"] + template["input"].format(instruction=qs["content"][i])
            else:
                p = prompts[i-1][j] + generations[i-1][j] + "\n\n" + template["input"].format(instruction=qs["content"][i])
            prompt.append(p)
        gen = llm.generate(prompt, sampling_params)

        prompts.append(prompt)
        generations.append(gen)

    outputs = []

    for i in range(len(prompts[0])):
        output_format = {
            "model": args.model_id,
            "benchmark": args.benchmark_type,
            "input": [pm[i] for pm in prompts],
            "output": [gen[i] for gen in generations]
        }
        outputs.append(output_format)

    # prompt_1 = [template["system"] + template["input"].format(instruction=ques["content"][0]) for ques in questions]
    # generation = llm.generate(prompt_1, sampling_params)
    # output_1 = [output.outputs[0].text for output in generation]
    # prompt_2 = [pm_1 + op_1 + "\n\n" + template["input"].format(instruction=ques["content"][1]) for pm_1, op_1, ques in zip(prompt_1, output_1, questions)]
    # generation = llm.generate(prompt_2, sampling_params)
    # output_2 = [output.outputs[0].text for output in generation]

    # for pm_1, op_1, pm_2, op_2 in zip(prompt_1, output_1, prompt_2, output_2):
    #     output_format = {
    #         "model": args.model_id,
    #         "benchmark": "mt-bench",
    #         "turn_1_prompt": pm_1,
    #         "turn_1_output": op_1,
    #         "turn_2_prompt": pm_2,
    #         "turn_2_output": op_2
    #     }
    #     outputs.append(output_format)
    
    return outputs

def run_eval(args):
    questions = load_benchmark(args.dataset_path, args.benchmark_type)
    template = load_template(args.prompt_path, args.prompt_type)
    outputs = get_model_outputs(
        args,
        questions,
        template
    )
    
    with open(args.output_path + args.model_id + "_result.json", "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    args = args_parse()

    run_eval(args)