# MergeEval
Merged model evaluation on MT-Bench, arena-hard, FLASK, and AlpacaEval

## Setup
```
cd MergeEval
pip install -r requirements.txt
```

## Run
```
python scripts/gen_answer.py \
    --model_name MODEL_CARD \
    --model_id MODEL_ID \
    --prompt_type PROMPT_TYPE \
    --benchmark_type BENCHMARK_TYPE
```
