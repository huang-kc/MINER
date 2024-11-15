import ast
import pickle
from pathlib import Path
import pandas as pd

def count_layer_num(input_list):
    from collections import defaultdict
    result = defaultdict(list)

    for key, value in input_list:
        result[key].append(value)

    result = dict(sorted(result.items()))
    num_result = {key: len(result[key]) for key in result.keys()}
    return result, num_result

output_path = Path('/data/kaichen/MINER/outputs/')
# output_path = Path('/data/kaichen/radar_onellm/modality_specific/demo_outputs')

def get_outputs(file_str=None):
    # import pdb
    # pdb.set_trace()
    for file in output_path.rglob('*.csv'):
        # print(file)
        contain_flag = True
        if file_str is not None:
            for Str in file_str:
                if Str not in str(file):
                    contain_flag = False
                    break
            if not contain_flag:
                continue
        
        try:
            df = pd.read_csv(file, on_bad_lines='warn')
        except Exception as e:
            print(file)
            print(e)
            print("wrong")
            exit()
            
        print(len(df), str(file.stem).split('--'))
        print(file)
        ret = {"correct": [], "bleu": [], "sbert_similarity": [], "cider": [], "wrr": []}
        for index, row in df.iterrows():
            for key in ret.keys():
                if key in row:
                    ret[key].append(row[key])
        if len(ret["correct"]) != 0:
            print("correct", sum(ret["correct"]) / len(ret["correct"]))
        elif len(ret["wrr"]) != 0:
            print("wrr", sum(ret["wrr"]) / len(ret["wrr"]))
        else:
            for key in ["bleu", "sbert_similarity", "cider"]:
                dic = {'max': [], 'min': [], 'mean': []}
                for i in range(len(df)):
                    single_dic = ast.literal_eval(ret[key][i])
                    for key2 in dic.keys():
                        dic[key2].append(single_dic[key2])
                for key2 in dic.keys():
                    print(key, key2, sum(dic[key2]) / len(df))
        print("")
        
def check_masks(file_str=None):
    for file in output_path.rglob('*mask.npy'):
        contain_flag = False
        if file_str is not None:
            for Str in file_str:
                if Str in str(file):
                    contain_flag = True
                    break
            if not contain_flag:
                continue

        print(file)
        with open(file, 'rb') as f:
            mask_dic = pickle.load(f)
            for modal in mask_dic.keys():
                print(modal, len(mask_dic[modal]))
                print(count_layer_num(mask_dic[modal])[1])
        print("")

# get_outputs(["layer_uniform"])
# get_outputs(["LA_MU"])
# get_outputs(["/vocal_sound/"])

# "prompt": {6: "prompt"},
    # "special_text_as_one": {2: "special_text", 3: "image", 4: "video", 5: "audio"},
    # "special_text_separate"

# get_outputs(["/coco_caption/"])
get_outputs()






# import pickle

# path1 = "/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_audio/mmlu/ISM/ISM_start0_end5000.npy"
# path2 = "/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_audio/mmlu/ISM/ISM_start5001_end10001.npy"
# path3 = "/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_audio/mmlu/ISM/ISM_start10001_end14042.npy"

# with open(path1, "rb") as f:
#     h1 = pickle.load(f)
    
# with open(path2, "rb") as f:
#     h2 = pickle.load(f)

# with open(path3, "rb") as f:
#     h3 = pickle.load(f)
# import pdb
# pdb.set_trace()

# with open("/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_audio/mmlu/ISM/ISM.npy", "wb") as f:
#     pickle.dump((h1[0]+h2[0]+h3[0], h1[1]+h2[1]+h3[1]), f)
# # a = 1

# from pathlib import Path
# vl_path = Path("/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_vl/sum_ISM")
# audio_path = Path("/data/kaichen/radar_onellm/modality_specific/outputs/qwen2_audio/sum_ISM")
# import os

# for file in audio_path.rglob('*'):
#     if file.is_dir():
#         sh_str = f"python main.py --mllm qwen2_audio --mode 3 --dataset mmlu --sum_ISM_path {file.name}"
#         os.system(sh_str)
#         # exit()
#         # print(file.name)
    
    