"""main python file"""
import os
import argparse
import sys
import yaml
import pandas as pd
from utils.dataset import DatasetProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "qwen2_audio"], help="the mllm to be analyzed")
    parser.add_argument('--dataset', type=str, default='text_vqa', choices=cfg["ALL_DATASETS"])
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")

    parser.add_argument('--mode', type=float, choices=[0,1,1.5,2,3], default=2)
    # mode 0: infer normally
    # mode 1: generate Impotance Score Matrix (ISM), stage1

    # stage1: generate ISM
    parser.add_argument('--sample_num', type=int, default=-1, help="number of samples, -1 means all samples")
    parser.add_argument('--sample_start_ind', type=int, default=0)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils import func as uf

    uf.set_seed(2024)
    args.sample_num = cfg["ALL_SAMPLE_NUMS"].get(args.dataset, args.sample_num) if args.sample_num == -1 else args.sample_num
    args.sample_num_start_from = 0
    
    # range(args.sample_start_ind, sargs.ample_end_ind)，左闭右开
    if args.sample_start_ind + args.sample_num < cfg["ALL_SAMPLE_NUMS"][args.dataset]:
        args.sample_end_ind = args.sample_start_ind + args.sample_num
    else:
        args.sample_end_ind = cfg["ALL_SAMPLE_NUMS"][args.dataset]
    # args.sample_str = f'start{args.sample_start_ind}_end{args.sample_end_ind}'

    args.demo_prefix = 'demo_' if args.demo else ''
    args.mllm_path = f"{args.demo_prefix}outputs/{args.mllm}"
    args.mllm_dataset_path = f"{args.mllm_path}/{args.dataset}"
    args.mllm_dataset_ISM_path = f"{args.mllm_dataset_path}/ISM"
    
    # create folder and initialize csv file 
    if args.mode in [1, 3]:
        base_folder_path = args.mllm_dataset_path
        args.csv_name = "origin"
        # args.csv_path = f"{base_folder_path}/{args.csv_name}_{args.sample_str}.csv"
        args.csv_path = f"{base_folder_path}/{args.csv_name}.csv"

        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)
            uf.initialize_csv(f'/{args.csv_name}.csv', args)
            
        elif not os.path.exists(args.csv_path):
            uf.initialize_csv(f'/{args.csv_name}.csv', args)        
        
        else: # csv file exist, check whether need to resume
            df = pd.read_csv(args.csv_path)
            if len(df) != args.sample_num:
                args.sample_start_ind = len(df)
                uf.initialize_csv(f'/{args.csv_name}.csv', args)
            else:
                print("Condition met, exiting the program...")
                sys.exit(0)
        
    dataset = DatasetProcessor(args) # also contains initialization of mllm
    for index in range(args.sample_start_ind, args.sample_end_ind):
        csv_line = dataset.infer(index)
        uf.handle_output(args, csv_line)

if __name__ == "__main__":
    main()