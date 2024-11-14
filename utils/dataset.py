import json
import os
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.crop import crop
from models import Qwen2_VL, Qwen2_Audio

def check_acc(response, ans):
    """
    Check if one of the answers appears in response
    """
    out = response.lower()
    for item in ans:
        if item.lower() in out:
            return True
    return False

class TextEvaluation:
    """
    evaluation between predicted sentence and ground truth sentences
    """
    def __init__(self):
        self.sbert_model = SentenceTransformer('/data/kaichen/data/paraphrase-MiniLM-L6-v2')

    def calculate_bleu(self, input_sentence, reference_sentences):
        """
        calculate BLEU score
        """
        input_tokens = input_sentence.split()
        reference_tokens_list = [ref.split() for ref in reference_sentences]
        smoothing_function = SmoothingFunction().method1

        bleu_scores = [
            sentence_bleu([ref], input_tokens, smoothing_function=smoothing_function) 
            for ref in reference_tokens_list
        ]
        result = {
            'max': max(bleu_scores),
            'min': min(bleu_scores),
            'mean': np.mean(bleu_scores),
            'scores': bleu_scores
        }
        return result

    def calculate_sbert_similarity(self, input_sentence, reference_sentences):
        """
        compute Sentence-BERT similarity
        """
        input_embedding = self.sbert_model.encode(input_sentence, convert_to_tensor=True)
        reference_embeddings = self.sbert_model.encode(reference_sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(input_embedding, reference_embeddings).cpu().numpy().flatten().tolist()
        result = {
            'max': max(cosine_scores),
            'min': min(cosine_scores),
            'mean': np.mean(cosine_scores),
            'scores': cosine_scores
        }
        return result

    def calculate_tfidf(self, captions):
        """
        compute TF-IDF
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(captions)
        return tfidf_matrix, vectorizer

    def calculate_cider(self, input_sentence, reference_sentences):
        """
        compute score of CIDEr
        """
        references = reference_sentences.copy()
        references.append(input_sentence)

        tfidf_matrix, _ = self.calculate_tfidf(references)

        sim_matrix = cosine_similarity(tfidf_matrix)
        cider_scores = sim_matrix[-1, :-1].tolist()
        result = {
            'max': max(cider_scores),
            'min': min(cider_scores),
            'mean': np.mean(cider_scores),
            'scores': cider_scores
        }
        return result

    def evaluate(self, input_sentence, reference_sentences):
        """
        compute all metrics
        """
        results = {
            'bleu': self.calculate_bleu(input_sentence, reference_sentences),
            'sbert_similarity': self.calculate_sbert_similarity(input_sentence, reference_sentences),
            'cider': self.calculate_cider(input_sentence, reference_sentences)
        }
        return results

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

class MSVD(Dataset):
    """
    MSVD_QA
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.datas = json.load(open('/data/kaichen/data/MSVD/test_qa.json', "r", encoding='utf-8'))
        map_ids =[x.strip().split(' ') for x in open('/data/kaichen/data/MSVD/youtube_mapping.txt', "r", encoding='utf-8').readlines()]
        self.id_to_video_ids = {x[1]:x[0] for x in map_ids}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = 'vid'+str(data['video_id'])
        video_name = self.id_to_video_ids[video_id] + '.avi'
        image_path = os.path.join("/data/kaichen/data/MSVD/YouTubeClips", video_name)
        question = data['question'] + '\nAnswer the question using a single word or phrase.'
        ret_data = {
            'text': question,
            'video': image_path
        }
        answer = data['answer']
        return ret_data, answer
    
    def infer(self, index):
        data, answer = self[index]
        response = self.model.infer(data)
        cor = check_acc(response, [answer])
        csv_line = {"index": index, "text": data["text"], "video": data["video"],
                    "answer": response, "label": answer, "correct": cor}
        return csv_line

class TextVQA(Dataset):
    """
    Text_VQA
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.datas = json.load(open("/data/kaichen/data/TextVQA/val/TextVQA_0.5.1_val.json", "r", encoding='utf-8'))['data']
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        vqa = self.datas[index]
        data = {
            'text': vqa['question'],
            'img': f"/data/kaichen/data/TextVQA/val/train_images/{vqa['image_id']}.jpg"
        }
        answer = vqa['answers']
        return data, answer
    
    def infer(self, index):
        data, answer = self[index]
        response = self.model.infer(data)
        
        cor = check_acc(response, answer)
        csv_line = {"index": index, "text": data["text"], "img": data["img"], 
                    "answer": response, "label": answer, "correct": cor}
        return csv_line

class Libri(Dataset):
    """
    LibriSpeech
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.paths, self.labels = self.load_libri()
        self.prompt = "Please transcribe the following audio directly into plain text without any additional explanations, prefixes, or descriptions. Only output the transcription of the spoken content in the audio."
        
    def load_libri(self):
        LIBRISPEECH_DIR = '/data/kaichen/data/LibriSpeech/test-clean'
        audio_data = []
        transcriptions = []
        for speaker_id in os.listdir(LIBRISPEECH_DIR):
            speaker_dir = os.path.join(LIBRISPEECH_DIR, speaker_id)
            if os.path.isdir(speaker_dir):
                for chapter_id in os.listdir(speaker_dir):
                    chapter_dir = os.path.join(speaker_dir, chapter_id)
                    if os.path.isdir(chapter_dir):
                        transcription_file = os.path.join(chapter_dir, f'{speaker_id}-{chapter_id}.trans.txt')
                        if os.path.exists(transcription_file):
                            with open(transcription_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    parts = line.strip().split(' ', 1)
                                    if len(parts) == 2:
                                        audio_filename, transcription = parts
                                        audio_filepath = os.path.join(chapter_dir, f'{audio_filename}.flac')
                                        audio_data.append(audio_filepath)
                                        transcriptions.append(transcription)
        return audio_data, transcriptions
        
    def remove_prefix(self, hyp, ref):
        ref = ref.lower().strip()
        hyp = hyp.lower().strip()
        if ref in hyp:
            return ref
        else:
            return hyp
        
    def wer(self, ref, hyp):
        """
        Word Error Rate (WER) = 1 - Word Recognition Rate(WRR)
        ref :ground truth
        hyp :hypothesis
        """
        # remove unrelated prefix
        hyp = self.remove_prefix(hyp, ref)

        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()

        # initialize edit distance matrix
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint8)

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        wer_score = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return wer_score
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        audio_path = self.paths[index]
        answer = self.labels[index]
        data = {
            "text": self.prompt,
            "audio": audio_path
        }
        return data, answer

    def infer(self, index):
        data, answer = self[index]
        response = self.model.infer(data)
        csv_line = {"index": index, "text": data["text"], "audio": data["audio"], 
                    "answer": response, "label": answer, "wrr": 1 - self.wer(answer, response)}
        return csv_line
        
class Vocal(Dataset):
    """
    VocalSound
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.file_lst = json.load(open("/data/kaichen/data/vocal_sound_release_16k/datafiles/all.json", "r", encoding='utf-8'))['data']
        self.old_path_str = "/data/sls/scratch/yuangong/vocalsound2/data/vs_processed/data_16k/"
        self.new_path_str = "/data/kaichen/data/vocal_sound_release_16k/audio_16k/"
        self.label_dict = {
            '/m/01j3sz': 'Laughter',
            '/m/07plz5l': 'Sigh',
            '/m/01b_21': 'Cough',
            '/m/0dl9sf8': 'Throat clearing',
            '/m/01hsr_': 'Sneeze',
            '/m/07ppn3j': 'Sniff'
        }
        self.prompt = """You are a sound classification model. Your task is to classify a given audio sample into one of the following categories based on its content:
                      1. Laughter
                      2. Sigh
                      3. Cough
                      4. Throat clearing
                      5. Sneeze
                      6. Sniff
                      Please analyze the audio sample and provide the corresponding category name.
                      """
        self.prompt = re.sub(r'\s+', ' ', self.prompt.replace("\n", "").strip())
    
    def __len__(self):
        return len(self.file_lst)
    
    def __getitem__(self, index):
        item = self.file_lst[index]
        audio_path, answer = item['wav'].replace(self.old_path_str, self.new_path_str), item['labels']
        answer = self.label_dict[answer]
        data = {
            "text": self.prompt,
            "audio": audio_path
        }
        return data, answer

    def infer(self, index):
        data, answer = self[index]
        response = self.model.infer(data)
        csv_line = {"index": index, "text": data["text"], "audio": data["audio"], 
                    "answer": response, "label": answer, "correct": (answer in response)}
        return csv_line

class MMLU(Dataset):
    """
    MMLU, require sequential access
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        subjects = sorted(f[:-9] for f in os.listdir('/data/kaichen/data/mmlu_data/test') if f.endswith('_test.csv'))
        self.current_subject = None
        self.current_subject_count = 0
        self.choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        self.mmlu_data = pd.read_parquet("/data/kaichen/data/mmlu_data/mmlu-test-all.parquet", engine='pyarrow')

        self.dev_test_dict = {}
        for subject in subjects:
            dev_df = pd.read_csv(f"/data/kaichen/data/mmlu_data/dev/{subject}_dev.csv", header=None)[:5]
            test_df = pd.read_csv(f"/data/kaichen/data/mmlu_data/test/{subject}_test.csv", header=None)
            self.dev_test_dict[subject] = {'dev': dev_df, 'test': test_df}
    
    def format_example(self, df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += f"\n{choices[j]}. {df.iloc[idx, j+1]}"
        prompt += "\nAnswer:"
        if include_answer:
            prompt += f" {df.iloc[idx, k + 1]}\n\n"
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = f"The following are multiple choice questions (with answers) about {self.format_subject(subject)}.\n\n"
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    def __len__(self):
        return len(self.mmlu_data)
    
    def __getitem__(self, index):
        case = self.mmlu_data.iloc[index]
        subject, answer = case['subject'], self.choices[case['answer']]
        
        # track the current subject to save
        if subject != self.current_subject:
            self.current_subject = subject
            self.current_subject_count = 0
        else:
            self.current_subject_count += 1
            
        k = 5
        prompt_end = self.format_example(self.dev_test_dict[subject]['test'], self.current_subject_count, include_answer=False)
        train_prompt = self.gen_prompt(self.dev_test_dict[subject]['dev'], subject, k)
        prompt = train_prompt + prompt_end
        
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = self.gen_prompt(self.dev_test_dict[subject]['dev'], subject, k)
            prompt = train_prompt + prompt_end

        data = {
            'text': prompt
        }
        return data, answer, subject, self.current_subject_count

    def infer(self, index):
        data, answer, subject, current_subject_count = self[index]
        response = self.model.infer(data)
        cor = False if len(response) == 0 else response[0] == answer
        csv_line = {"index": index, "dataset name": subject, "sub-index": current_subject_count, "text": data["text"],
                    "answer": response, "label": answer, "correct": cor}
        return csv_line

class COCO_caption(Dataset):
    """
    COCO Caption
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.coco_data = self.karpathy_coco_test()
        self.evaluator = TextEvaluation()
        self.coco_prompt = """
                           Generate a caption for the image in one short sentence, similar to these examples from the COCO dataset: 
                           1. A man with a red helmet on a small moped on a dirt road.
                           2. Man riding a motor bike on a dirt road on the countryside.
                           3. A man riding on the back of a motorcycle.
                           4. A man in a red shirt and a red hat is on a motorcycle on a hill side.
                           Now, describe the image.
                           """
        self.coco_prompt = re.sub(r'\s+', ' ', self.coco_prompt.replace("\n", "").strip())
    
    def karpathy_coco_test(self):
        """
        generate karpathy coco testset, 5000 images with multiple captions in total
        """
        coco = COCO('/data/kaichen/data/coco/annotations_trainval2014/captions_val2014.json')
        root_img_path = '/data/kaichen/data/coco/val2014/'
        old_imgIds = coco.getImgIds()
        with open('/data/kaichen/data/coco/karparthy_split2014/coco_test.txt', 'r', encoding='utf-8') as f:
            kar_test = [line.strip() for line in f]

        imgIds = []
        for img_id in old_imgIds:
            filename = coco.loadImgs(img_id)[0]['file_name']
            if filename in kar_test:
                img_path = root_img_path + filename
                imgIds.append((img_id, img_path))

        ground_truths = {img_id: {'img': img_path, 'anns': []} for (img_id, img_path) in imgIds}

        for img_id in imgIds:
            img_id = img_id[0]
            annIds = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                ground_truths[img_id]['anns'].append(ann['caption'])
        return list(ground_truths.items())
    
    def __len__(self):
        return len(self.coco_data)
    
    def __getitem__(self, index):
        value = self.coco_data[index][1]
        answer = value['anns']
        data = {
            'text': self.coco_prompt,
            'img': value['img']
        }  
        return data, answer

    def infer(self, index):
        data, answer = self[index]
        response = self.model.infer(data)
        ret = self.evaluator.evaluate(response, answer)
        csv_line = {"index": index, "text": data["text"], "img": data["img"], 
                    "answer": response, "label": answer, "bleu": ret["bleu"], 
                    "cider": ret["cider"], "sbert_similarity": ret["sbert_similarity"]}
        return csv_line

class Test(Dataset):
    """
    test dataset
    """
    def __init__(self) -> None:
        super().__init__()
        self.datas = [
            {'text': "describe this image", 'img': "scripts/test.jpg"},
            {'text': "What are the words in the picture", 'img': "scripts/random.jpg"},
            {'text': "What's that sound?", 'audio': "scripts/glass-breaking.mp3"}
        ]
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index], None

class DatasetProcessor:
    def __init__(self, args):        
        # create mllm model
        if args.mllm == 'qwen2_vl':
            model = Qwen2_VL(args)
        elif args.mllm == 'qwen2_audio':
            model = Qwen2_Audio(args)
        else:
            pass # add more models
        
        self.datasets = {
            'text_vqa': TextVQA,
            'coco_caption': COCO_caption,
            'mmlu': MMLU,
            'msvd_qa': MSVD,
            'libri': Libri,
            'vocal_sound': Vocal,
            'test': Test
        }
        self.dataset = self.datasets.get(args.dataset, lambda model: None)(model)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def infer(self, index):
        return self.dataset.infer(index)