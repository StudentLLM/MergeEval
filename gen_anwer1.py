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

def get_model_outputs(
    model_name,
    model_id,
    benchmark_type,
    prompt_path,
    prompt_type,
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

    # 지금은 1st turn 하나 생성하고 바로 이어서 2nd turn 생성
    outputs = []
    
    prompt_1 = [template["system"] + template["input"].format(instruction=ques["content"][0]) for ques in questions]
    generation = llm.generate(prompt_1, sampling_params)
    output_1 = [output.outputs[0].text for output in generation]
    prompt_2 = [pm_1 + op_1 + "\n\n" + template["input"].format(instruction=ques["content"][1]) for pm_1, op_1, ques in zip(prompt_1, output_1, questions)]
    generation = llm.generate(prompt_2, sampling_params)
    output_2 = [output.outputs[0].text for output in generation]

    for pm_1, op_1, pm_2, op_2 in zip(prompt_1, output_1, prompt_2, output_2):
        output_format = {
            "model": model_id,
            "benchmark": "mt-bench",
            "turn_1_prompt": pm_1,
            "turn_1_output": op_1,
            "turn_2_prompt": pm_2,
            "turn_2_output": op_2
        }
        outputs.append(output_format)

    
    # for question in tqdm(questions):
    #     choices = []
    #     for i in range(num_choices):
    #         torch.manual_seed(i)
    #         prompt = template["system"]

    #         turns = []
    #         for j in range(len(question["content"])):
    #             q = question["content"][j]
    #             prompt += template["input"].format(instruction=q)
    #             output = llm.generate(prompt, sampling_params)
    #             prompt += output[0].outputs[0].text
    #             turns.append(output)
    #         choices.append({"index": i, "outputs": outputs})
    #     outputs.append(choices)

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
    prompt_path,
    prompt_type,
    dataset_path,
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
        prompt_path,
        prompt_type,
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

    # ans_json = [{
    #     "model_id": model_id,
    #     "category": question["category"],
    #     "content": question["content"],
    #     "output": outputs[i]
    # } for i, question in enumerate(questions)]
    
    with open(output_path + model_id + "_result.json", "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    args = args_parse()

    run_eval(
        args.model_name,
        args.model_id,
        args.benchmark_type,
        args.prompt_path,
        args.prompt_type,
        args.dataset_path,
        args.num_choices,
        args.dtype,
        args.temperature,
        args.max_tokens,
        args.min_tokens,
        args.top_p,
        args.repetition_penalty,
        args.output_path
    )
