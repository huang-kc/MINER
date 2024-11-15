# switch args.demo to 1 before run the demo program
# conda activate radar

# generate ISM per sample
python ISM_per_sample.py --dataset coco_caption --sample_num 22 --demo 1
python ISM_per_sample.py --dataset mmlu --sample_num 26 --demo 1
python ISM_per_sample.py --dataset text_vqa --sample_num 29 --demo 1

# aggregate ISM to get sum_ISM
python ISM_aggregate.py --mllm qwen2_vl --load_ISM_sample_num -1 -1 -1 -1 -1 -1 --demo 1

# generate and apply mask
python ISM_gen_and_apply.py --mllm qwen2_vl --dataset mmlu --sample_num 22 --sum_ISM_path text_vqa29_coco_caption22_mmlu26 --demo 1