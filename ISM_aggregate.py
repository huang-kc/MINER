"""main python file"""
import os
import argparse
import sys
import pickle
import yaml

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "qwen2_audio"])
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")
    parser.add_argument('--load_ISM_sample_num', type=int, nargs='+', default=[-1,-1,-1,-1,-1,-1], help="ISM_x.npy for [text_vqa,coco_caption,mmlu,msvd_qa,libri,vocal_sound], 0 means not use, -1 means ISM.npy")
    
    args = parser.parse_args()
    mllm_path = f"{'demo_' if args.demo else ''}outputs/{args.mllm}"
    sum_ISM_path = f"{mllm_path}/sum_ISM/"

    # read ISM of all possible modalities
    ISM_list = []
    for dataset, ISM_sample_num in zip(cfg["ALL_DATASETS"][1:], args.load_ISM_sample_num):
        if ISM_sample_num == 0: continue
        ISM_path = f"{mllm_path}/{dataset}/ISM/{'ISM.npy' if ISM_sample_num == -1 else f'ISM_{ISM_sample_num}.npy'}"
        if os.path.exists(ISM_path):
            sample_num, ISM = pickle.load(open(ISM_path, "rb"))
            ISM /= sample_num # normalize to [0,1]
            assert ISM.max() <= 1
            ISM_list.append(ISM)
            sum_ISM_path += f'{dataset}{sample_num}_'
            print(f"load ISM from {ISM_path} with {sample_num} samples.")

    if len(ISM_list) == 0:
        print("Oops! ISM file not exists, load nothing!")
        sys.exit(0)

    # Normalize based on the different frequencies of modalities in the datasets.
    sum_ISM = sum(ISM_list)
    occurrence_times = {}
    for ind, modality in enumerate(cfg["ALL_MODALITIES"]):
        count = sum(ISM[:, ind].sum() != 0 for ISM in ISM_list)
        occurrence_times[modality] = int(count)
        if count != 0:
            sum_ISM[:, ind] /= count # normalize to [0,1]
    print(f'occurrence times of different modalities: {occurrence_times}')
    
    # create folder for sum_ISM and 3 kinds splits
    new_path = sum_ISM_path[:-1]
    os.makedirs(new_path, exist_ok=True)
    file_path = os.path.join(new_path, 'sum_ISM.npy')
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump(sum_ISM, f)
    else:
        print(f"{file_path} already exists!")

    # 3 different splits: prompt.npy, special_text_separate.npy, special_text_as_one.npy
    # delete modalities never occur and have conflit with current split
    for split_type, index_modality_map in cfg["ALL_MODILITY_SPLIT_TYPES"].items():
        modality_list, ISM_index_list = [], []
        for index, modality in index_modality_map.items():
            if sum_ISM[:, index].sum() == 0: continue
            modality_list.append(modality)
            ISM_index_list.append(index)
        with open(f"{new_path}/{split_type}.npy", "wb") as f:
            pickle.dump((modality_list, sum_ISM[:, ISM_index_list]), f)
                
if __name__ == "__main__":
    main()