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
    parser.add_argument("--model_id", type=str
    parser.add_argument("--prompt_path", type=str, default="prompt/")
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--dataset_path", type=str, default="data/")
    parser.add_argument("--benchmark_type", type=str)
    parser.add_argument("--output_path", type=str, default="output/")
    parset.add_argument("--num_choices", type=int, default=1)
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
        template = json.loads(tem_file)
        return template

def get_model_outputs(
    model_name,
    model_id,
    benchmark_type,
    questions,
    template,
    num_choices,
    dtype,
    temperature,
    max_tokens,
    min_tokens,
    top_p,
    repetition_penalty,
    output_path
):
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype=dtype
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    outputs = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            prompt = template["system"]

            turns = []
            for j in range(len(question["turns"]:
                q = question["turns"][j]
                prompt += template["input"].format(instruction=q)
                output = llm.generate(prompt, sampling_params)
                prompt += output.outputs[0].text
                turns.append(output)
            choices.append({"index": i, "outputs": outputs})
        outputs.append(choices)

        # os.makedirs(output_path, exist_ok=True)
        # output_dir = "/".join([output_path, model_id, "(", benchmark_type, ")", "_result.json"])
        # with open(output_dir, "a") as fout:
        #     ans_json = {
        #         "model_id": model_id,
        #         "category": question["category"],
        #         "content": question["content"],
        #         "choices": choices
        #     }
        #     json.dump(ans_json, fout, intent=4)
    
    return outputs

def run_eval(
    model_name,
    model_id,
    benchmark_type,
    num_choices,
    dtype,
    temperature,
    max_tokens,
    min_tokens,
    top_p,
    repetition_penalty,
    output_path
):
    questions = load_benchmark(dataset_path, benchmark_type)
    template = load_template(prompt_path, prompt_type)

    outputs = get_model_outputs(
        model_name,
        model_id,
        benchmark_type,
        questions,
        template,
        num_choices,
        dtype,
        temperature,
        max_tokens,
        min_tokens,
        top_p,
        repetition_penalty,
        output_path
    )

    ans_json = [{
        "model_id": model_id,
        "category": question["category"],
        "content": question["content"],
        "output": outputs[i]
    } for i, question in enumerate(questions)]
    
    with open("/".join([output_path, model_id, "(", benchmark_type, ")", "_result.json"]), "w") as f:
        json.dump(ans_json, f, indent=4)

if __name__ == "__main__":
    args = args_parse()

    run_eval(
        args.model_name,
        args.model_id,
        args.benchmark_type,
        args.num_choices,
        args.dtype,
        args.temperature,
        args.max_tokens,
        args.min_tokens,
        args.top_p,
        args.repetition_penalty,
        args.output_path
    )
