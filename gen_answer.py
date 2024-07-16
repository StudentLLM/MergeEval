from vllm import LLM, SamplingParams
from datasets import load_dataset
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

    prompts = []
    generations = []

    num_turns = questions[0]["content"]
    for i in range(len(num_turns)):
        prompt = []
        for j, qs in enumerate(questions):
            if i == 0:
                p = template["system"] + template["input"].format(instruction=qs["content"][i])
            else:
                p = prompts[i-1][j] + generations[i-1][j] + "\n\n" + template["input"].format(instruction=qs["content"][i])
            prompt.append(p)
        gen = llm.generate(prompt, sampling_params)
        gen = [g.outputs[0].text for g in gen]
        
        prompts.append(prompt)
        generations.append(gen)

    outputs = []

    for i in range(len(questions)):
        output_format = {
            "model": args.model_id,
            "benchmark": args.benchmark_type,
            "template": args.prompt_type,
            "input": [pm[i] for pm in prompts],
            "output": [gen[i] for gen in generations]
        }
        outputs.append(output_format)
    
    return outputs

def run_eval(args):
    questions = load_benchmark(args.dataset_path, args.benchmark_type)
    template = load_template(args.prompt_path, args.prompt_type)
    outputs = get_model_outputs(
        args,
        questions,
        template
    )

    os.makedirs(args.output_path, exist_ok=True)
    with open(args.output_path + args.model_id + "_" + args.benchmark_type + "_result.json", "w") as f:
        json.dump(outputs, f, default=str, indent=4)

if __name__ == "__main__":
    args = args_parse()

    run_eval(args)
