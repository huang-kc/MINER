"""utils functions to assist other python files"""
import random
import os
import csv
import re
import sys
import pickle
import itertools
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
import torch
import torch.nn.functional as F

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def split_mask_prefix(tasks, file_name):
    """
    split the mask prefix into "task" and "sample_ratio"
    """
    for task in tasks:
        if file_name.startswith(task):
            params = file_name[len(task):]
            params = params.lstrip('_')
            return task, params if params else None
    return None, None

def append_suffix_to_filename(file_path, suffix="_cp"):
    """
    add suffix
    """
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{suffix}{ext}"
    new_file_path = os.path.join(directory, new_filename)
    return new_file_path

def is_cp_suffix(file_path):
    """
    check the cp suffix
    """
    _, filename = os.path.split(file_path)
    name, _ = os.path.splitext(filename)
    return name.endswith('_cp')

def set_seed(seed):
    """
    set seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def min_max_normalize(tensor):
    """
    min max normalization
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def save_ISM_with_token_mask_in_one_layer(old_tensor, modality_info, args):
    """
    save ISM of each neuron in one layer
    """    
    index, modality, layer_index = modality_info
    mask = args.input_modality_masks.get(modality)
    if mask is None or mask.sum() == 0:
        return
    tensor = old_tensor.clone()[mask, :]
    flat_tensor = tensor.view(-1, tensor.shape[-1])

    importance_dict = {}
    # prob
    positive_counts = (tensor > 0).sum(dim=0)
    total_tokens = tensor.size(0)
    prob_val = positive_counts / total_tokens
    importance_dict['prob'] = min_max_normalize(prob_val)
    # mean
    mean_val = torch.mean(flat_tensor, dim=0)
    importance_dict['mean'] = min_max_normalize(mean_val)
    # max
    max_val, _ = torch.max(flat_tensor, dim=0)
    importance_dict['max'] = min_max_normalize(max_val)
    # attn
    mask = mask.squeeze(0)
    attn_score = args.attn_score.squeeze(0)
    attn_score = attn_score[mask, :][:, mask]
    # attn_k
    attn_k = F.softmax(attn_score, dim=0) # dim0, softmax on column / key
    attn_val_k = torch.matmul(attn_k.T, flat_tensor)
    attn_val_k = torch.sum(attn_val_k, dim=0)
    importance_dict['attn_k'] = min_max_normalize(attn_val_k)
    # attn_q
    attn_q = F.softmax(attn_score, dim=1) # dim1, softmax on row / query
    attn_val_q = torch.matmul(attn_q, flat_tensor)
    attn_val_q = torch.sum(attn_val_q, dim=0)
    importance_dict['attn_q'] = min_max_normalize(attn_val_q)

    # save ISM of layer neurons to modality tokens
    for ind, metric_type in enumerate(cfg["ALL_IMPORTANCE_METRIC_TYPES"]):
        args.ISM_of_one_sample[ind, index, layer_index] = importance_dict[metric_type]

def find_topK_values(data_dict, ret, K, return_flag=False): # no overlap
    if return_flag:
        ret = {'-': []}
    combined_tensor = sum(data_dict.values())
    _, topk_indices = torch.topk(combined_tensor.flatten(), K)
    
    indices_2d = [torch.div(index, combined_tensor.shape[1], rounding_mode='floor').item() for index in topk_indices], \
                 [index % combined_tensor.shape[1] for index in topk_indices]
        
    for i in range(K):
        row, col = indices_2d[0][i], indices_2d[1][i]
        max_tensor_name = None
        max_value = float('-inf')
        
        for name, tensor in data_dict.items():
            if tensor[row, col] > max_value:
                max_value = tensor[row, col]
                max_tensor_name = name
        ret[max_tensor_name].append((int(row), int(col)))
    return ret if return_flag else None

def analyse_modality_specific_neurons(input_dict, L):
    modalities = list(input_dict.keys())
    modality_combinations = []
    for i in range(1, len(modalities) + 1):
        modality_combinations.extend(combinations(modalities, i))

    columns = ['-'.join(combo) for combo in modality_combinations]
    df = pd.DataFrame(0, index=list(range(L)) + ['total'], columns=columns)

    for modality, neurons in input_dict.items():
        for layer in range(L):
            neuron_count = sum(1 for l, _ in neurons if l == layer)
            df.at[layer, modality] = neuron_count

    for combo in modality_combinations:
        combo_name = '-'.join(combo)
        if len(combo) > 1:
            for layer in range(L):
                neurons_per_layer = [set(n for l, n in input_dict[modality] if l == layer) for modality in combo]
                intersection = set.intersection(*neurons_per_layer) if neurons_per_layer else set()
                df.at[layer, combo_name] = len(intersection)
    df.loc['total'] = df.sum(axis=0)
    return df

def select_modality_neurons_from_importance_scores(sum_ISM, mask_index):
    """
    input: ISM, select_ratio, importance_metric_weights, select_strategy
    output: {'text': [(0, 22),...]}
    """
    _, select_ratio, importance_metric_weights, select_strategy = mask_index
    importance_metric_weights = [float(i) for i in importance_metric_weights.split('_')]
    
    modalities, ISM = sum_ISM
    importance_metric_weights = torch.tensor(importance_metric_weights).view((5,1,1,1)).to(ISM.device)
    weighted_ISM = (ISM * importance_metric_weights).sum(dim=0)
    
    weighted_ISM_dict = {}
    M, L, N = len(modalities), ISM.shape[-2], ISM.shape[-1]
    K = int(select_ratio * L * N) # 5304
    
    modality_specific_neurons = {modality: [] for modality in (modalities + ["random"])}
    for index, modality in enumerate(modalities):
        weighted_ISM_dict[modality] = weighted_ISM[index]
        
    if select_strategy == "random":
        modality_specific_neurons["random"] = [(random.randint(0, L - 1), random.randint(0, N - 1)) for _ in range(K)]
    elif select_strategy == "adaptive":
        find_topK_values(weighted_ISM_dict, modality_specific_neurons, K)
    elif select_strategy == "LA_MU":
        for modal, modal_scores in weighted_ISM_dict.items():
            find_topK_values({modal: modal_scores}, modality_specific_neurons, K // M)           
    elif select_strategy == "LU_MA":
        Map = {i: modalities[i] for i in range(M)}
        for layer in range(L):
            sub_dict = {'-': torch.stack([v[layer] for v in weighted_ISM_dict.values()])}
            ret = find_topK_values(sub_dict, modality_specific_neurons, K // L, return_flag=True)['-']
            for (modal_ind, neuron_ind) in ret:
                modality_specific_neurons[Map[modal_ind]].append((layer, neuron_ind))            
    elif select_strategy == "uniform":
        for layer in range(L):
            matrix = torch.zeros(L, N).to(ISM.device)
            for modal, modal_scores in weighted_ISM_dict.items():
                matrix[layer] = modal_scores[layer]
                find_topK_values({modal: matrix}, modality_specific_neurons, K // (M * L))
    
    modality_specific_neurons = {k: v for k, v in modality_specific_neurons.items() if v}
    analyse_df = analyse_modality_specific_neurons(modality_specific_neurons, L)
    return modality_specific_neurons, analyse_df

def create_activation_hook(layer_index, args):
    """
    hook for activation of mlp neurons in llm
    """
    def activation_hook(module, input, output):
        # compute and save ISM
        if args.save_ISM and (output.shape[:-1] == args.prompt_mask_shape or output.shape[-2] > 1):
            if args.dataset in ['libri', 'vocal_sound'] and output.shape[:-1] != args.prompt_mask_shape:
                # if data has audio modality, need to update all masks first
                _, num_tokens = args.prompt_mask_shape
                pad_token_num = output.shape[-2] - num_tokens
                assert args.input_modality_masks['audio'].sum() == 1
                _, start_ind = torch.nonzero(args.input_modality_masks['audio'] == 1, as_tuple=True)
                start_ind += 1
                
                zero_value = torch.zeros((1, pad_token_num), dtype=torch.bool).to(args.device)
                one_value = torch.ones((1, pad_token_num), dtype=torch.bool).to(args.device)
                for modality, mask in args.input_modality_masks.items():
                    if modality == 'audio':
                        args.input_modality_masks[modality] = torch.cat((mask[:, :start_ind], one_value, mask[:, start_ind:]), dim=1)
                    else:
                        args.input_modality_masks[modality] = torch.cat((mask[:, :start_ind], zero_value, mask[:, start_ind:]), dim=1)
                args.prompt_mask_shape = args.input_modality_masks['audio'].shape
                
            for index, modality in enumerate(cfg["ALL_MODALITIES"]):
                save_ISM_with_token_mask_in_one_layer(output, (index, modality, layer_index), args)
        
        # apply the generated mask
        if args.apply_mask:
            if args.mask_modalities[0] == "all_modalities":
                wanna_mask_modalities = list(args.modality_specific_neurons.keys())
            else:
                supported_modalities = list(args.modality_specific_neurons.keys())
                wanna_mask_modalities = args.mask_modalities
                if not set(wanna_mask_modalities).issubset(set(supported_modalities)):
                    print(f"Unsupported mask modalities: {wanna_mask_modalities}, {args.modality_split_type} only support {supported_modalities}!")
                    sys.exit(0)
            
            all_positions = []
            for modality, neurons in args.modality_specific_neurons.items():
                if modality not in wanna_mask_modalities:
                    continue
                all_positions += neurons
                
            layer_positions = [t[1] for t in all_positions if t[0] == layer_index]

            if args.complementary_mask:
                com_positions = list(set(range(args.hidden_size)) - set(layer_positions))
                output[..., com_positions] = output.min() if args.deactivation_val == -1 else args.deactivation_val            
            else:
                output[..., layer_positions] = output.min() if args.deactivation_val == -1 else args.deactivation_val            
        return output
    return activation_hook

def save_attn_matrix(attention_matrix, name, args):
    attention_matrix = attention_matrix.to(torch.float32).squeeze(0).cpu().numpy()
    # token_names = [i for i in range(len(attention_matrix[-1]))]
    token_names = list(args.input_ids.squeeze(0).cpu().numpy())
    plt.figure(figsize=(40, 30))

    sns.heatmap(attention_matrix, xticklabels=token_names, yticklabels=token_names, cmap='viridis', annot=False, fmt=".2f")
    plt.title("Attention Heatmap")
    plt.xlabel("Tokens (Key)")
    plt.ylabel("Tokens (Query)")
    plt.savefig(name, dpi=300, bbox_inches='tight')

def create_attention_hook(layer_index, args):
    """
    hook for attention layer in llm
    """
    def attn_hook(module, input, output):
        args.attn_score = module.attn_score
        # save_attn_matrix(args.attn_score, f'{layer_index}.png', args)
        return output
    return attn_hook

def initialize_csv(args):
    args.csv_fieldnames = ["index", "dataset name", "sub-index", "text", "img", "video", "audio", "answer", "label"]
    args.csv_file = open(args.csv_path, mode='a+', newline='', encoding='utf-8')
    def initialize_csv_writer(args, add_lst):
        args.csv_fieldnames += add_lst
        args.writer = csv.DictWriter(args.csv_file, fieldnames=args.csv_fieldnames)
        args.csv_file.seek(0, 0)
        if args.csv_file.read(1) == '':
            args.csv_file.seek(0, 0)
            args.writer.writeheader()
    if args.dataset in ["text_vqa", "mmlu", "msvd_qa", "vocal_sound"]:
        initialize_csv_writer(args, ["correct"])
    elif args.dataset == "coco_caption":
        initialize_csv_writer(args, ["bleu", "sbert_similarity", "cider"])
    elif args.dataset == "libri":
        initialize_csv_writer(args, ["wrr"])
    
def generate_mask_of_modality_specific_neurons(
    args,
    sum_ISM_file_name=None,
    modality_split_type=None,
    select_ratio=None,
    importance_metric_weights=None,
    select_strategy=None,
):
    """
    All parameters are optional. If none are specified, all possible values will be iterated.
    input: (
        sum_ISM_file_name: text_vqa5000_coco_caption5000_mmlu14042,
        modality_split_type,
        select_ratio,
        importance_metric_weights (list of length 5),
        select_strategy,
    )
    output: (
        sum_ISM_path/masks.npy is updated,
        last modality_specific_neuron_mask is saved in args.modality_specific_neuron_mask
    )
    """
    all_sum_ISM_paths = []
    for subfolder in Path(args.sum_ISM_path).rglob('*'):
        if subfolder.is_dir() and re.search(r'(' + '|'.join(cfg["ALL_DATASETS"]) + r')\d+', str(subfolder)):
            if sum_ISM_file_name is not None and subfolder.name != sum_ISM_file_name:
                continue
            all_sum_ISM_paths.append(subfolder.name)
    
    for temp_sum_ISM_path in all_sum_ISM_paths:
        sum_ISM_path = f"{args.sum_ISM_path}/{temp_sum_ISM_path}"
        masks_path = f"{sum_ISM_path}/masks.npy"
        if os.path.exists(masks_path): # update incrementally
            try:
                with open(masks_path, "rb") as f:
                    mask_dict = pickle.load(f)
            except Exception as e:
                print(e)
                print(masks_path)
                sys.exit()
        else:
            mask_dict = {}
        
        # iterate all possible combination of parameters
        cartesian_product = list(itertools.product(
            list(cfg["ALL_MODILITY_SPLIT_TYPES"].keys()),
            cfg["ALL_SELECT_RATIO"],
            cfg["ALL_IMPORTANCE_METRIC_WEIGHTS"],
            cfg["ALL_SELECT_STRATEGIES"],
        ))
        for item in cartesian_product:
            if (modality_split_type is not None and item[0] != modality_split_type) or \
                (select_ratio is not None and item[1] != select_ratio) or \
                (importance_metric_weights is not None and item[2] != importance_metric_weights) or \
                (select_strategy is not None and item[3] != select_strategy):
                continue
            # These parameters can uniquely determine a mask.
            mask_index = (item[0], item[1], "_".join(map(str, item[2])), item[3])
            
            if mask_dict.get(mask_index) is not None:
                args.modality_specific_neurons = mask_dict.get(mask_index)[0]
                print(f"mask of {temp_sum_ISM_path}, {mask_index} already exists!")
                continue

            with open(f"{sum_ISM_path}/{item[0]}.npy", "rb") as f:
                ISM_temp = pickle.load(f)
            ret = select_modality_neurons_from_importance_scores(ISM_temp, mask_index)
            mask_dict[mask_index] = ret # (mask, df)
            args.modality_specific_neurons = ret[0]
            print(f"successfully save mask of {temp_sum_ISM_path}, {mask_index}!")
    
        with open(masks_path, "wb") as f:
            pickle.dump(mask_dict, f)

    if not hasattr(args, 'modality_specific_neurons'):
        print("Oops! something wrong, fail to generate mask!")
        inputs = [modality_split_type, importance_metric_weights, select_ratio, select_strategy]
        choices = [list(cfg["ALL_MODILITY_SPLIT_TYPES"].keys()), cfg["ALL_IMPORTANCE_METRIC_WEIGHTS"], cfg["ALL_SELECT_RATIO"], cfg["ALL_SELECT_STRATEGIES"]]

        # check if inputs in choices
        violations = []
        for i, (value, lst) in enumerate(zip(inputs, choices)):
            if value not in lst:
                violations.append(f"Value {value} at index {i} not in list {lst}")
        print(violations)
        sys.exit(0)