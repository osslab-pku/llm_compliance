from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tqdm
from collections import  defaultdict
import re
from transformers import AutoTokenizer
import pandas as pd
from utils import chunks
from joblib import Parallel, delayed
from pygments.lexers import PythonLexer
from pygments import lex
from datasketch import MinHash, MinHashLSH
MODEL_PATH = "WizardLMTeam/WizardCoder-15B-V1.0"  # replace it with the local path

LICENSE_TERMS: pd.DataFrame = pd.read_csv("terms.csv")
def get_5grams(s):
    s = split_code(s)
    words = s.split()
    return [' '.join(words[i:i+5]) for i in range(len(words) - 4)]

def compute_minhash_from_file(file_content):
    m = MinHash(num_perm=256)
    for gram in get_5grams(file_content):
        m.update(gram.encode('utf8'))
    return m
def split_code(code):
    tokens = list(lex(code, PythonLexer()))
    values = [value for ttype, value in tokens]
    return (" ").join(values)

def get_license_type(license: str) -> str:
    df = LICENSE_TERMS
    df['license'] = df['license'].apply(lambda x: x.lower())
    this_term = df[df["license"] == license].to_dict(orient="records")
    if this_term:
        this_term = this_term[0]
        copyleft = this_term["copyleft"]
        if copyleft == 0:
            return "Permissive"
        elif copyleft == 1 or copyleft == 2:
            return "Weak Copyleft"
        else:
            return "Strong Copyleft"
    else:
        return "Unknown"

def _cal_blue(res_random,samples):
    res_bleu=defaultdict(list)
    res_random = dict(res_random)
    for id in res_random:
        reference_text_list=[]
        for generated_text in res_random[id]:
            generated_text = split_code(generated_text).split()
            reference_text_list.append(split_code(samples[id]['body']).split())
            res_bleu[id].append(sentence_bleu(reference_text_list, generated_text, smoothing_function = SmoothingFunction().method4))
    return res_bleu

def cal_blue(res_random,samples):
    res_bleu=defaultdict(list)
    chunk_lst = chunks(list(res_random.items()), 200)
    res = Parallel(n_jobs=40)(
        delayed(_cal_blue)(task,samples)
        for task in tqdm.tqdm(chunk_lst, total=len(res_random)//200)
    )
    for i in res:
        res_bleu.update(i)
    return res_bleu

def _cal_jaccard(res_random,samples):
    jaccard=defaultdict(list)
    res_random = dict(res_random)
    for id in tqdm.tqdm(res_random):
        for generated_text in res_random[id]:
            
            generated_text_minhash = compute_minhash_from_file(generated_text)
            reference_text_minhash = compute_minhash_from_file(samples[id]['body'])
        
            jaccard[id].append(generated_text_minhash.jaccard(reference_text_minhash))
    return jaccard

def cal_jaccard(res_random,samples):
    jaccard=defaultdict(list)
    chunk_lst = chunks(list(res_random.items()), 200)
    res = Parallel(n_jobs=40)(
        delayed(_cal_jaccard)(task,samples)
        for task in tqdm.tqdm(chunk_lst, total=len(res_random)//200)
    )
    for i in res:
        jaccard.update(i)
    return jaccard

def levenshtein_distance_star(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def get_edit_distance(candidate, reference, tokenizer, length = 4096):
    reference = reference.strip()
    c = tokenizer.encode(reference, add_special_tokens=False)[:length] if length else tokenizer.encode(reference, add_special_tokens=False)
    candidate = candidate.strip()
    r = tokenizer.encode(candidate, add_special_tokens=False)[:length] if length else tokenizer.encode(candidate, add_special_tokens=False)
    return levenshtein_distance_star(c, r)/max(len(c), len(r)) if max(len(c), len(r)) > 0 else 1

def _cal_edit_distance_sim(res_random,samples):
    res_random = dict(res_random)
    res_edit_distance=defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    for id in tqdm.tqdm(res_random):
        for generated_text in res_random[id]:
            res_edit_distance[id].append(1-get_edit_distance(generated_text, samples[id]['body'], tokenizer)) #return the similarity
    return res_edit_distance

def cal_edit_distance_sim(res_random,samples):
    res_edit_distance=defaultdict(list)
    chunk_lst = chunks(list(res_random.items()), 200)
    res = Parallel(n_jobs=40)(
        delayed(_cal_edit_distance_sim)(task,samples)
        for task in tqdm.tqdm(chunk_lst, total=len(res_random)//200)
    )
    for i in res:
        res_edit_distance.update(i)
    return res_edit_distance

def get_comment(prompt,body,answer):
    body_comments = set(re.findall(r'#(.*)', body))
    answer_comments = set(re.findall(r'#(.*)', answer))
    intersection = body_comments.intersection(answer_comments)
    intersection = list(map(lambda x:x.strip(),intersection))
    
    intersection = list(filter(lambda x: x not in prompt, intersection))
    return len(intersection)
            




    