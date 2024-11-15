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
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "qwen2_audio"])
    parser.add_argument('--dataset', type=str, default='text_vqa', choices=cfg["ALL_DATASETS"])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")

    parser.add_argument('--sample_num', type=int, default=-1, help="number of samples, -1 means all samples")
    parser.add_argument('--sample_start_ind', type=int, default=0)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils import func as uf

    uf.set_seed(2024)
    args.sample_num = cfg["ALL_SAMPLE_NUMS"].get(args.dataset, args.sample_num) if args.sample_num == -1 else args.sample_num
    args.sample_num_start_from = 0
    
    if args.sample_start_ind + args.sample_num < cfg["ALL_SAMPLE_NUMS"][args.dataset]:
        args.sample_end_ind = args.sample_start_ind + args.sample_num
    else:
        args.sample_end_ind = cfg["ALL_SAMPLE_NUMS"][args.dataset]

    # create folder and initialize csv file 
    mllm_dataset_path = f"{'demo_' if args.demo else ''}outputs/{args.mllm}/{args.dataset}"    
    args.csv_path = f"{mllm_dataset_path}/origin.csv"
    args.mllm_dataset_ISM_path = f"{mllm_dataset_path}/ISM"

    if not os.path.exists(mllm_dataset_path):
        os.makedirs(mllm_dataset_path)

    if not os.path.exists(args.csv_path):
        uf.initialize_csv(args)
    else:
        print("wrong!")
        sys.exit(0)
    # else: # csv file exist, check whether need to resume
    #     df = pd.read_csv(args.csv_path)
    #     if len(df) != args.sample_num:
    #         args.sample_start_ind = len(df)
    #         uf.initialize_csv(f'/origin.csv', args)
    #     else:
    #         print("Condition met, exiting the program...")
    #         sys.exit(0)
    
    dataset = DatasetProcessor(args, save_ISM=True) # also contains initialization of mllm
    for index in range(args.sample_start_ind, args.sample_end_ind):
        csv_line = dataset.infer(index)
        args.writer.writerow(csv_line)
        args.csv_file.flush()

if __name__ == "__main__":
    main()