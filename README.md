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

## Test SparseQJL on Llama
Params:
* `qjl`: Boolean flag denoting whether or not to apply QJL. Default: `False`
* `sparsity`: Float between 0 and 1 denoting \% uniform sparsity with SparseGPT. Default: `0.0`
* `wbits`: Int denoting the bit-width for weight quantization. We suggest using a value of `4`. Default: `16` (No quant)
* `dtype`: String denoting standard datatype of model. Options are `float16` and `float32`. Default: `float16`.

Note:
* `meta-llama/Llama-2-7b-hf`: takes about 20 minutes to run with pruning
* `meta-llama/Llama-2-13b-hf`: takes about 35 minutes to run with pruning
```
python llama_sparseqjl.py --model_name "meta-llama/Llama-2-7b-hf" \
    --qjl False \
    --sparsity 0.5 \
    --wbits 4 \
    --dtype "float16"
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
| LLaMA-2-13b           | 4.58           | 6.57    | 52.16    |
| **SparseQJL**         |                |         |          |
| LLaMA-2-7b            | 6.09           | 9.43    | 491.04   |
| LLaMA-2-13b           | 5.33           | 8.33    | 85.23    |
