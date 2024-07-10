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
