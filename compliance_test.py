import tqdm
from utils import chunks
import json
import os
from joblib import Parallel, delayed
from openai import OpenAI
import argparse
import multiprocessing
import jsonlines
from collections import defaultdict
from extract_main import extract
from analysis import cal_blue, cal_jaccard, cal_edit_distance_sim, get_comment, get_license_type
from check_license import find_license_by_keywords

#api_key: str = open("key_path", "r").readline().strip()  # replace with your key path
#base_url: str = ""  # replace with your base url
benchmark: dict = json.load(open(f'benchmark.json'))

def chat(chunk: list, m: str, lock) -> dict:
    res: dict = {}
    file_name: str = m.split("/")[-1]
    if os.path.exists(f"res/{file_name}.jsonl"):
        with jsonlines.open(f"res/{file_name}.jsonl", "r") as jsonl:
            for line in jsonl:
                res[line["id"]] = line["generated_text"]
    client: OpenAI = OpenAI(
        base_url=base_url, 
        api_key=api_key
    )
    for q in chunk:
        if q[0] in res and res[q[0]] != "":
            continue    
        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Complete the following Python function, don't say any natural language:\n" + q[1]["header"],
                    }
                ],
                model=m,
                temperature=0,
                max_tokens=4096
            )
            if len(completion.choices) > 0:
                for choice in completion.choices:
                    content: str = choice.message.content
                    res[q[0]] = content
                    with lock:
                        with jsonlines.open(f"res/{file_name}.jsonl", "a") as jsonl:
                            jsonl.write({"id": q[0], "generated_text": content})

        except Exception as e:
            print(f"Error: {e}")

    return res

def query(m: str) -> dict:
    chunk_lst: list = list(chunks(list(benchmark.items()), 100))
    completion_res: dict = {}
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    results: list = Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(chat)(lst, m, lock) for lst in tqdm.tqdm(chunk_lst, total=len(chunk_lst))
    )
    
    for r in results:
        completion_res.update(r)
    
    file_name: str = m.split("/")[-1]
    json.dump(completion_res, open(f"res/{file_name}.json", "w"))
    return completion_res
    
def extract_answer(res: dict, m: str) -> dict:
    file_name: str = m.split("/")[-1]
    res_extracted: defaultdict = defaultdict(list)
   
    for i in res:
        if i not in benchmark:
            continue
        code: str = res[i]
        res_extracted[i].append(extract(code, benchmark[i]["signature"].split("(")[0]).strip("\n"))
    json.dump(res_extracted, open(f'res/{file_name}_extracted.json', 'w'))
    return res_extracted

def calculate_sim(res: dict, m: str) -> tuple:
    file_name: str = m.split("/")[-1]

    bleu_score: dict = cal_blue(res, benchmark)
    json.dump(bleu_score, open(f'res/{file_name}_bleu.json', 'w'))
    jacard_score: dict = cal_jaccard(res, benchmark)
    json.dump(jacard_score, open(f'res/{file_name}_jaccard.json', 'w'))
    edit_distance_sim: dict = cal_edit_distance_sim(res, benchmark)
    json.dump(edit_distance_sim, open(f'res/{file_name}_edit_distance_sim.json', 'w'))
    return bleu_score, jacard_score, edit_distance_sim

def detect_striking_sim(res: dict, bleu: dict, jaccard: dict, edit_distance: dict, m: str) -> dict:
    file_name: str = m.split("/")[-1]
    striking_sim: dict = {}
    for i in res:
        max_sim: float = max([bleu[i][0], jaccard[i][0], edit_distance[i][0]]) 
        if max_sim > 0.6:
            comment_cnt: int = get_comment(benchmark[i]['header'], benchmark[i]['body'], res[i][0])
            if comment_cnt > 0:
                striking_sim[i] = {'bleu': bleu[i][0], 'jaccard': jaccard[i][0], 'edit_distance': edit_distance[i][0],
                                   'comment_cnt': comment_cnt, 'max_sim': max_sim, 'header': benchmark[i]['header'],
                                   'body': benchmark[i]['body'], 'answer': res[i][0], "license": benchmark[i]['license'],
                                   "author": benchmark[i]['author'], "project": benchmark[i]['project'],
                                   "signature": benchmark[i]['signature'], "license_type": benchmark[i]['license_type'],
                                   "init_header": benchmark[i]['init_header'], "docstring": benchmark[i]['docstring']}
                striking_sim[i]['license_type'] = get_license_type(striking_sim[i]['license'])
    json.dump(striking_sim, open(f'res/{file_name}_striking_sim.json', 'w'))
    return striking_sim

def ask_for_license(striking_sim: dict, answer: dict, m: str) -> None:
    file_name: str = m.split('/')[-1]
    res: dict = {}
    client: OpenAI = OpenAI(
        base_url=base_url,  # replace with your base url
        api_key=api_key 
    )
    for i in tqdm.tqdm(striking_sim):
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Complete the following Python function, don't say any natural language:\n" + benchmark[i]["header"],
                },
                {
                    "role": "assistant",
                    "content": answer[i],
                },
                {
                    "role": "user",
                    "content": "Give me the license or copyright information of the code above. Just give me the license name."
                }
            ],
            model=m,
            temperature=0,
        )
        if len(completion.choices) > 0:
            for choice in completion.choices:
                content: str = choice.message.content
                res[i] = content
    json.dump(res, open(f'res/{file_name}_license.json', 'w'))
    return res

def calculate_lico(striking_sim: int, A_perm: float, A_copy: float, w1: int = 1, w2: int = 2, w3: int = 4, total: int = 4187) -> float:
    if A_perm is None:
        A_perm = 1
    if A_copy is None:
        A_copy = 1
    N: float = striking_sim / total

    lico: float = (w1 * (1 - N) + w2 * A_perm + w3 * A_copy) / (w1 + w2 + w3)
    lico = round(lico, 3)
    return lico  

def calculate_res(m: str , license_res:dict) -> dict:
    file_name: str = m.split('/')[-1]
    answer: dict = {}
    result: dict = {}
    c: int = 0
    permissive: int = 0
    pc: int = 0
    copyleft: int = 0
    cc: int = 0
    for i in license_res:
        license_answered: str = find_license_by_keywords(license_res[i])
        if benchmark[i]["license_type"] == "Permissive":
            permissive += 1
        else:
            copyleft += 1
        answer[i] = (benchmark[i]["license"], benchmark[i]["license_type"], license_answered, 1 if license_answered == benchmark[i]["license"] else 0)
        if license_answered == benchmark[i]["license"]:
            c += 1
        if license_answered == benchmark[i]["license"] and benchmark[i]["license_type"] == "Permissive":
            pc += 1
        if license_answered == benchmark[i]["license"] and benchmark[i]["license_type"] != "Permissive":
            cc += 1
    json.dump(answer, open(f'res/{file_name}_license_answer.json', 'w'))
    
    if len(license_res) > 0:
        result["striking_sim"] = {
            "count": len(license_res),
            "percentage": f"{len(license_res)/len(benchmark)*100:.2f}%"
        }
        result["accuracy"] = c/len(license_res)
        result["permissive"] = {
            "count": permissive,
            "accuracy": pc/permissive if permissive > 0 else None
        }
        result["copyleft"] = {
            "count": copyleft,
            "accuracy": cc/copyleft if copyleft > 0 else None
        }
        result["lico"] = calculate_lico(
            len(license_res),
            pc/permissive if permissive > 0 else None,
            cc/copyleft if copyleft > 0 else None
        )
    
    return result

   
if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt-4o", type=str
    )
    args: argparse.Namespace = parser.parse_args()
    if args.model:
        comp_res: dict = query(args.model)
       
        extracted_res: dict = extract_answer(comp_res, args.model)
        bleu, jaccard, edit_distance = calculate_sim(extracted_res, args.model)
        striking_sim: dict = detect_striking_sim(extracted_res, bleu, jaccard, edit_distance, args.model)
        license_res = ask_for_license(striking_sim, comp_res, args.model)
        res = calculate_res(args.model, license_res)

        print(res)