from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import torch
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--benchmark_type", type=str)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    
    return parser.parse_args()

def load_benchmark():
    questions = []
    with open("", "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return

def get_prompt():
    return

def get_model_output(
    model_name
    dtype,
    temperature,
    max_tokens,
    min_tokens,
    top_p,
    repetition_penalty
):
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        min_tokens=50,
        top_p=1.0,
        repetition_penalty=1.03
    )

    outputs = llm.generate(question_set, sampling_params)
    return outputs

def run_eval():
    return

if __name__ == "__main__":
    args = args_parse()



    result_dataset = {
        "prompt": [],
        "output": []
    }
    
    for output in outputs:
        result_dataset["prompt"].append(output.prompt)
        result_dataset["output"].append(output.outputs[0].text)
        
    with open("/".join([args.output_path, args.model_id + "_result.json"]), "w") as f:
        json.dump(result_dataset, f, indent=4)
