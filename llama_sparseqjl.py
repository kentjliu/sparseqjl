import os
import argparse
import random
import time
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from model.llama2_utils_qjl import QJLSketch
from model.llama2_qjl import LlamaForCausalLM_QJL

import torch.nn as nn

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

from utils.sparsegpt import *
from utils.modelutils import *
from utils.quant import *

from torch.cuda.amp import autocast

    
def get_tokenizer(model):
    '''
    Fetch respective model tokenizer from HuggingFace
    '''
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad_token
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    '''
    Prepare Wikitext2 dataset
    '''
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
    
def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    '''
    Prepare PTB dataset
    '''
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    '''
    Prepare C4 dataset
    '''
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def seed_everything(seed):
    '''
    For reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    '''
    Fetch respective loaders
    '''
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)


def setup_model_and_tokenizer(
        model_name,
        qjl=False,
        dtype=torch.float16,
        key_quantization_bits=256,
        key_quantization_bits_initial_layers=512,
        initial_layers_count=15,
        outlier_count_general=8,
        outlier_count_initial_layers=8,
        value_quantization_bits=2,
        group_size=32,
        buffer_size=128,
):
    '''
    Fetch respective llama models according to args
    '''
    model_name = 'meta-llama/Meta-Llama-3-8B'
    device = 'cuda'
    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True

    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        tokenizer_type='llama'
    )

    model = None

    if qjl:
        config.attention_dropout = 0.0
        config.key_quantization_bits = key_quantization_bits
        config.key_quantization_bits_initial_layers = key_quantization_bits_initial_layers
        config.initial_layers_count = initial_layers_count

        config.outlier_count_general = outlier_count_general
        config.outlier_count_initial_layers = outlier_count_initial_layers

        config.value_quantization_bits = value_quantization_bits
        config.group_size = group_size
        config.buffer_size = buffer_size

        generator = torch.Generator(device=torch.device(device))

        config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
        config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
                                                rot=True,
                                                rng=generator)

        config.use_flash = True

        model = LlamaForCausalLM_QJL.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        print('vanilla model')
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    return model, tokenizer


def parse_args(args=None):
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--qjl', action='store_true')
    parser.add_argument('--wbits', type=int, default=16)
    parser.add_argument('--dtype', type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument('--key_quantization_bits', type=int, default=256)
    parser.add_argument('--key_quantization_bits_initial_layers', type=int, default=512)
    parser.add_argument('--initial_layers_count', type=int, default=15)
    parser.add_argument('--outlier_count_general', type=int, default=8)
    parser.add_argument('--outlier_count_initial_layers', type=int, default=8)
    parser.add_argument('--value_quantization_bits', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_data', type=int, default=150)
    parser.add_argument('--sparsity', type=float, default=0)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], default='c4',
        help='Where to extract calibration data from.'
    )
    return parser.parse_args(args)


def load_configurations(config_dir):
    '''
    Load dataset and model configurations
    '''
    with open(os.path.join(config_dir, 'dataset2maxlen.json')) as f:
        dataset2maxlen = json.load(f)
    with open(os.path.join(config_dir, 'dataset2prompt.json')) as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(config_dir, 'model2maxlen.json')) as f:
        model2maxlen = json.load(f)

    return dataset2maxlen, dataset2prompt, model2maxlen


@torch.no_grad()
def llama_sequential(model, dataloader, dev, sparsity=0.5, blocksize=128):
    '''
    SparseGPT implementation for sequential layer weight pruning
    '''
    print("Starting pruning...")
    model.seqlen = model.config.max_position_embeddings

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # batch_size = 8
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    # Class to intercept activations
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError  # Early exit

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']
    print("Ready to prune.")

    quantizers = {}
    for i, layer in enumerate(layers):
        print(f"Pruning layer {i}...")
        layer = layer.to(dev)
        full = find_layers(layer)
        
        # sequential = [list(full.keys())]
        sequential = [["self_attn.k_proj", "self_attn.v_proj"]]

        for names in sequential:
            subset = {n: full[n] for n in names}
            
            gpts = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            # Register hooks to gather input/output activations
            handles = []
            
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                
            for j in range(128):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                
            for h in handles:
                h.remove()
                
            for name in subset:
                print(i, name)
                print("Pruning ...")
                gpts[name].fasterprune(
                    sparsity=sparsity,
                    prunen=0,
                    prunem=0,
                    percdamp=0.01,
                    blocksize=blocksize
                )
                gpts[name].free()
                
        for j in range(128):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Pruning completed.")
    return quantizers


@torch.no_grad() 
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    """
    Evaluate the perplexity of a LLaMA model with explicit position_ids handling.
    """
    print("Evaluating ...")
    model.seqlen = model.config.max_position_embeddings  # Set sequence length dynamically

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    # Class to intercept and save activations
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # Replace the first layer with Catcher
    layers[0] = Catcher(layers[0])

    # Process input batches to capture activations
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # Restore original layer

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    # Process each layer sequentially
    for i in range(len(layers)):
        print(f"Processing layer {i}")
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] # inps[j]: torch.Size([4096, 4096])

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final model normalization and lm_head
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    # Calculate perplexity
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # Compute final perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.3f}")

    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


# @torch.no_grad()
# def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
#     print("Evaluating ...")
#     seqlen = 4096

#     testenc = testenc.input_ids
#     nsamples = testenc.numel() // seqlen

#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers

#     model.model.embed_tokens = model.model.embed_tokens.to(dev)
#     layers[0] = layers[0].to(dev)

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {"i": 0, "attention_mask": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, inp, **kwargs):
#             inps[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs["attention_mask"]
#             raise ValueError

#     layers[0] = Catcher(layers[0])
#     for i in range(nsamples):
#         batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
#         try:
#             model(batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module

#     layers[0] = layers[0].cpu()
#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache["attention_mask"]

#     for i in range(len(layers)):
#         print(i)
#         layer = layers[i].to(dev)

#         # if args.gmp:
#         #     subset = find_layers(layer)
#         #     for name in subset:
#         #         W = subset[name].weight.data
#         #         thresh = torch.sort(torch.abs(W.flatten()))[0][
#         #             int(W.numel() * args.sparsity)
#         #         ]
#         #         W.data[torch.abs(W.data) <= thresh] = 0

#         for j in range(nsamples):
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps

#     if model.model.norm is not None:
#         model.model.norm = model.model.norm.to(dev)
#     model.lm_head = model.lm_head.to(dev)

#     testenc = testenc.to(dev)
#     nlls = []
#     for i in range(nsamples):
#         hidden_states = inps[i].unsqueeze(0)
#         if model.model.norm is not None:
#             hidden_states = model.model.norm(hidden_states)
#         lm_logits = model.lm_head(hidden_states)
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(
#             shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
#         )
#         neg_log_likelihood = loss.float() * seqlen
#         nlls.append(neg_log_likelihood)
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
#     print(f"Perplexity: {ppl.item():3f}")
#     if log_wandb:
#         wandb.log({f"{dataset}/perplexity": ppl.item()})

#     model.config.use_cache = use_cache


def main(args):
    seed_everything(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model_qjl, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.qjl,
        dtype,
        args.key_quantization_bits,
        args.key_quantization_bits_initial_layers,
        args.initial_layers_count,
        args.outlier_count_general,
        args.outlier_count_initial_layers,
        args.value_quantization_bits,
        args.group_size,
        args.buffer_size,
    )
    print(f"Model and tokenizer for {args.model_name} are set up successfully.")

    model = model_qjl
    model.eval()
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=128, seed=0, model=args.model_name, seqlen=4096, 
    )
    
    if args.sparsity:
        tick = time.time()
        llama_sequential(model, dataloader, DEV, sparsity=args.sparsity, blocksize=args.blocksize)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)
    
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=args.model_name, seqlen=2048
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, False)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    