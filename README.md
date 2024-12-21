# hpml-final
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kentjliu/hpml-final/blob/main/demo.ipynb)

## Description
SparseQJL is a large language model compression approach which leverages mathematical properties of the 
Johnson-Lindenstrauss Lemma for KV cache quantization and combines that with SparseGPT-based weight
pruning. This hybrid strategy significantly improves memory efficiency and reduces the number of model parameters, enabling
the scalable deployment of large-scale transformer models with extended context windows while retaining reasonable accuracy.

NOTE: to run code, ensure you have a HuggingFace token with the proper permissions for the necessary models.
One way you can do this is by running `hugginface-cli login`. When prompted for your token, paste it.

## Outline
* `models`: contains code and helper files for qjl tranformers based on original implementations
* `qjl_kernel`: contains cuda kernels and python config code
* `utils`: contains helper files for evaluation scripts

## Setup
Build the QJL kernel:
```
git clone https://github.com/kentjliu/sparseqjl.git
cd sparseqjl
python qjl_kernel/setup.py build_ext --inplace --build-temp=./qjl_kernel/build --build-lib=./qjl_kernel
```

## Test SparseQJL on Llama

Params:

* `model_name`: String denoting HuggingFace Llama model path to test. Default: `meta-llama/Llama-2-7b-hf`
* `qjl`: Boolean flag denoting whether or not to apply QJL.
* `sparsity`: Float between 0 and 1 denoting \% uniform sparsity with SparseGPT. Default: `0.0`
* `wbits`: Int denoting the bit-width for weight quantization. We suggest using a value of `4`. Default: `16` (No quant)
* `dtype`: String denoting standard datatype of model. Options are `float16` and `float32`. Default: `float16`.

Note:
* `meta-llama/Llama-2-7b-hf`: takes about 20 minutes to run with pruning
* `meta-llama/Llama-2-13b-hf`: takes about 35 minutes to run with pruning

Example usage:
```
python llama_sparseqjl.py --model_name "meta-llama/Llama-2-7b-hf" \
    --qjl True \
    --sparsity 0.5 \
    --wbits 4 \
    --dtype "float16"
```

## Test SparseQJL on OPT
Params:

* `model_name`: String denoting HuggingFace OPT model path to test. Default: `facebook/opt-125m`
* `qjl`: Boolean flag denoting whether or not to apply QJL.
* `sparsity`: Float between 0 and 1 denoting \% uniform sparsity with SparseGPT. Default: `0.0`
* `wbits`: Int denoting the bit-width for weight quantization. We suggest using a value of `4`. Default: `16` (No quant)
* `dtype`: String denoting standard datatype of model. Options are `float16` and `float32`. Default: `float16`.

Note:

* Currently, our implementation of SparseQJL on OPT is still not 100\% refined, hence the extremely high perplexity scores. We will continue to resolve the issue and make updates to this repo. 
* When prompted to run custom code, answer `y` in the CLI

Example usage:
```
python opt_sparseqjl.py --model_name "facebook/opt-350m" \
    --qjl \
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
