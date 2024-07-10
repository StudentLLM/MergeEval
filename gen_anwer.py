from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import torch
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--benchmark_path", type=str)
    parser.add_argument("--output_path", type=str, default="output/")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()

    llm = LLM(
        model=args.model_name, 
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

    result_dataset = {
        "prompt": [],
        "output": []
    }
    
    for output in outputs:
        result_dataset["prompt"].append(output.prompt)
        result_dataset["output"].append(output.outputs[0].text)
        
    with open("/".join([args.output_path, model_type + "_result.json"]), "w") as f:
        json.dump(result_dataset, f, indent=4)
