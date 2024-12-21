import time

import torch
import torch.nn as nn

from utils.quant import *
from utils.sparsegpt import *
from utils.modelutils import *
from utils.datautils import *

from model.opt_utils import QJLSketch
from model.opt_qjl import OPTForCausalLM_JL_Kernel

from transformers import OPTConfig, AutoTokenizer, OPTForCausalLM

import argparse

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 

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
    Fetch respective OPT models according to args
    '''
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    device = 'cuda'
    config = OPTConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
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

        model = OPTForCausalLM_JL_Kernel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        model = OPTForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    model.seqlen = model.config.max_position_embeddings

    return model, tokenizer


def parse_args(args=None):
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="facebook/opt-125m")
    parser.add_argument('--qjl', type=bool, default=False)
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
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    return parser.parse_args(args)


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

# def get_opt(model):
#     import torch
#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip
#     from transformers import OPTForCausalLM
#     model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
#     model.seqlen = model.config.max_position_embeddings
#     # 
#     return model 

# def get_opt(model_name):
#     import torch
#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip

    

#     # from model.opt_modified import OPTForCausalLM_JL
#     # model = OPTForCausalLM_JL.from_pretrained(model_name)

#     # from model.opt_modified_2 import OPTForCausalLM2
    
#     from model.opt_qjl import OPTForCausalLM_JL_Kernel

#     from transformers import OPTConfig
#     device = 'cuda'
    
#     config = OPTConfig.from_pretrained(model_name)
#     config.attention_dropout = 0.0
#     config.key_quantization_bits = 256
#     config.key_quantization_bits_initial_layers = 512
#     config.initial_layers_count = 15

#     config.outlier_count_general = 8
#     config.outlier_count_initial_layers = 8

#     config.value_quantization_bits = 2
#     config.group_size = 32
#     config.buffer_size = 128


#     head_dim = config.hidden_size // config.num_attention_heads

#     generator = torch.Generator(device=torch.device(device))
#     config.qjl = QJLSketch(dim=(head_dim, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
#     config.qjl_initial_layers = QJLSketch(dim=(head_dim, config.key_quantization_bits_initial_layers), dim_outlier=128,
#                                               rot=True,
#                                               rng=generator)
#     config.use_flash = True

#     model = OPTForCausalLM_JL_Kernel.from_pretrained(
#         pretrained_model_name_or_path=model_name, 
#         torch_dtype='auto',
#         config=config,
#         cache_dir=None,
#         low_cpu_mem_usage=True,
#         device_map="auto"
#     )
    
#     model.seqlen = model.config.max_position_embeddings
#     return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
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
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, idx=i)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, idx=i)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, idx=i)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024
    print('MEMORY INFO')
    print(f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}")

    total_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024 / 1024  # Convert to GB
    print(f"Total Parameters: {total_params:,}")
    print(f"Memory Taken by Parameters: {param_memory:.4f} GB")

    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache



def main(args):
    seed_everything(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

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
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model_name, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model_name, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)


if __name__ == "__main__":
    args = parse_args()
    main(args)
