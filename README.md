# hpml-final
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kentjliu/hpml-final/blob/main/demo.ipynb)

## Description
SparseQJL is a large language model compression approach which leverages mathematical properties of the 
Johnson-Lindenstrauss Lemma for KV cache quantization and combines that with SparseGPT-based weight
pruning. This hybrid strategy significantly improves memory efficiency and reduces the number of model parameters, enabling
the scalable deployment of large-scale transformer models with extended context windows while retaining reasonable accuracy.

### Outline
TODO

## Installation
Set up the QJL kernel:
```
python qjl_kernel/setup.py build_ext --inplace
```

### Usage
`python opt_qjl.py facebook/opt-125m c4 --sparsity 0.5 --qjl_ratio 0.5`

```
python llama_sparseqjl.py --model_name "meta-llama/Llama-2-7b-hf" \
    --dtype "float16" \
    --key_quantization_bits 256 \
    --key_quantization_bits_initial_layers 512 \
    --initial_layers_count 15 \
    --outlier_count_general 8 \
    --outlier_count_initial_layers 8 \
    --value_quantization_bits 2 \
    --group_size 32 \
    --buffer_size 128 \
    --seed 42 \
    --dataset_name [dataset_name] \
    --n_data 150 \
    --sparse True \
    --sparsity 0.5 \
    --blocksize 128
```

## Results
The following table displays perplexities on selected NLP tasks
| **Model**             | **Wikitext-2** | **C4**  | **PTB**  |
|-----------------------|----------------|---------|----------|
| **Baseline Models**   |                |         |          |
| LLaMA-2-7b            | 8.71           | 23.52   | 133.24   |
| LLaMA-2-13b           | 7.68           | 16.90   | 180.15   |
| **SparseGPT (50% Uniform Sparsity)** | | | |
| LLaMA-2-7b            | 7.02           | 9.24    | 186.68   |
| LLaMA-2-13b           | 6.02           | 8.22    | 76.81    |
| **QJL**               |                |         |          |
| LLaMA-2-7b            | 5.12           | 7.10    | 40.58    |
| LLaMA-2-13b           | 4.58           | 6.57.   | 52.16    |
| **SparseQJL**         |                |         |          |
| LLaMA-2-7b            | 6.09           | 9.43    | 491.04   |
| LLaMA-2-13b           | 5.33           | 8.33    | 85.23    |