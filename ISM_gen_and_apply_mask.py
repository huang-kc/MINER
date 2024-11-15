import os
import argparse
import yaml
from utils.dataset import DatasetProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="A simple example script to demonstrate argparse.")    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "qwen2_audio"], help="the mllm to be analyzed")
    parser.add_argument('--dataset', type=str, default='text_vqa', choices=cfg["ALL_DATASETS"])
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")

    parser.add_argument('--sample_num', type=int, default=-1, help="number of samples, -1 means all samples")
    parser.add_argument('--sample_start_ind', type=int, default=0)

    # generate mask
    parser.add_argument('--sum_ISM_file_name', type=str, default="text_vqa5000_coco_caption5000_mmlu14042")
    parser.add_argument('--select_ratio', type=float, default=0.02, choices=cfg["ALL_SELECT_RATIO"], help="the ratio of selected neurons")
    parser.add_argument('--importance_metric_weights', default=[0,0.25,0.25,0.5,0], type=float, nargs='+', help="weights of different importance metric: [prob,mean,max,attn_k,attn_q], sum=1")
    parser.add_argument('--select_strategy', default="LA_MU", type=str, choices=cfg["ALL_SELECT_STRATEGIES"])
    # apply mask
    parser.add_argument('--modality_split_type', default="special_text_separate", type=str, choices=list(cfg["ALL_MODILITY_SPLIT_TYPES"].keys()))
    parser.add_argument('--mask_modalities', type=str, nargs='+', default=["all_modalities"], help="choose modalities want to mask, should be subset of MLLM_MODALITIES[args.dataset]")
    parser.add_argument('--deactivation_val', type=float, default=0, help="output value of a deactivated neuron, -1 means output.min()")
    parser.add_argument('--complementary_mask', type=int, default=0)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils.func import set_seed, initialize_csv, generate_mask_of_modality_specific_neurons
    set_seed(2024)
    args.sample_num = cfg["ALL_SAMPLE_NUMS"].get(args.dataset, args.sample_num) if args.sample_num == -1 else args.sample_num
    args.sample_num_start_from = 0
    
    if args.sample_start_ind + args.sample_num < cfg["ALL_SAMPLE_NUMS"][args.dataset]:
        args.sample_end_ind = args.sample_start_ind + args.sample_num
    else:
        args.sample_end_ind = cfg["ALL_SAMPLE_NUMS"][args.dataset]

    # create folder and initialize csv file 
    mllm_path = f"{'demo_' if args.demo else ''}outputs/{args.mllm}"
    base_folder_path = f"{mllm_path}/{args.dataset}/mask_csv"
    args.mllm_dataset_ISM_path = f"{mllm_path}/{args.dataset}/ISM"
    args.sum_ISM_path = f"{mllm_path}/sum_ISM"

    args.csv_path = f"{base_folder_path}/" \
                    f"{args.sum_ISM_file_name}--" \
                    f"{args.modality_split_type}--" \
                    f"{args.select_ratio}--" \
                    f"{'_'.join(map(str, args.importance_metric_weights))}--" \
                    f"{args.select_strategy}--" \
                    f"{'_'.join(args.mask_modalities)}--" \
                    f"{args.deactivation_val}--" \
                    f"com{args.complementary_mask}.csv"

    if not os.path.exists(base_folder_path):
        os.makedirs(base_folder_path)
        
    if not os.path.exists(args.csv_path):
        initialize_csv(args)        
    
    generate_mask_of_modality_specific_neurons(
        args,
        sum_ISM_file_name=args.sum_ISM_file_name,
        modality_split_type=args.modality_split_type,
        select_ratio=args.select_ratio,
        importance_metric_weights=args.importance_metric_weights,
        select_strategy=args.select_strategy,
    )        
    
    dataset = DatasetProcessor(args, apply_mask=True) # also contains initialization of mllm
    for index in range(args.sample_start_ind, args.sample_end_ind):
        csv_line = dataset.infer(index)
        args.writer.writerow(csv_line)
        args.csv_file.flush()

if __name__ == "__main__":
    main()