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
from analysis import cal_blue, cal_jaccard, cal_edit_distance_sim,get_comment,get_license_type
from check_license import find_license_by_keywords
api_key = open("key_path", "r").readline().strip()  # replace with your key path
base_url = "",  # replace with your base url
benchmark = json.load(open(f'benchmark.json'))

def chat(chunk,m,lock):
    res = {}
    file_name = m.split("/")[-1]
    if os.path.exists(f"{file_name}.jsonl"):
        with jsonlines.open(f"{file_name}.jsonl", "r") as jsonl:
            for line in jsonl:
                res[line["id"]] = line["generated_text"]
    client = OpenAI(
        base_url=base_url, # replace with your base url
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
                    content = choice.message.content
                    res[q[0]] = content
                    with lock:
                        with jsonlines.open(f"{file_name}.jsonl", "a") as jsonl:
                            jsonl.write({"id": q[0], "generated_text": content})

        except Exception as e:
            
            print(f"Error: {e}")

    return res

def query(m):
    chunk_lst = list(chunks(list(benchmark.items()), 100))
    completion_res = {}
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    results = Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(chat)(lst, m, lock) for lst in tqdm.tqdm(chunk_lst, total=len(chunk_lst))
    )
    
    for r in results:
        completion_res.update(r)
    
    file_name = m.split("/")[-1]
    json.dump(completion_res, open(f"{file_name}.json", "w"))
    return completion_res
    
def extrect_answer(res,m):
    file_name = m.split("/")[-1]
    res_extracted = defaultdict(list)
   
    for i in res:
        if i not in benchmark:
            continue
        code = res[i]
        res_extracted[i].append(extract(code,benchmark[i]["signature"].split("(")[0]).strip("\n"))
    json.dump(res_extracted,open(f'{file_name}_extracted.json','w'))
    return res_extracted


def calculate_sim(res,m):
    file_name = m.split("/")[-1]

    bleu_score = cal_blue(res, benchmark)
    json.dump(bleu_score, open(f'{file_name}_bleu.json', 'w'))
    jacard_score = cal_jaccard(res, benchmark)
    json.dump(jacard_score, open(f'{file_name}_jaccard.json', 'w'))
    edit_distance_sim = cal_edit_distance_sim(res, benchmark)
    json.dump(edit_distance_sim, open(f'{file_name}_edit_distance_sim.json', 'w'))
    return bleu_score, jacard_score, edit_distance_sim

def detect_striking_sim(res,bleu,jaccard,edit_distance,m):
    file_name = m.split("/")[-1]
    striking_sim ={}
    for i in res:
        
        max_sim = max([bleu[i][0],jaccard[i][0],edit_distance[i][0]]) 
        if max_sim > 0.6:
            comment_cnt = get_comment(benchmark[i]['header'],benchmark[i]['body'],res[i][0])
            if comment_cnt > 0:
                striking_sim[i] = {'bleu':bleu[i][0],'jaccard':jaccard[i][0],'edit_distance':edit_distance[i][0],'comment_cnt':comment_cnt,'max_sim':max_sim,'header':benchmark[i]['header'],'body':benchmark[i]['body'],'answer':res[i][0],"license":benchmark[i]['license'],"author":benchmark[i]['author'],"project":benchmark[i]['project'],"signature":benchmark[i]['signature'],"license_type":benchmark[i]['license_type'],"init_header":benchmark[i]['init_header'],"docstring":benchmark[i]['docstring']}
                striking_sim[i]['license_type'] = get_license_type(striking_sim[i]['license'])
    json.dump(striking_sim,open(f'{file_name}_striking_sim.json','w'))
    return striking_sim
def ask_for_license(striking_sim,answer,m):
    file_name = m.split('/')[-1]
    res = {}
    client = OpenAI(
        base_url=base_url, # replace with your base url
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
            #stream=True
        )
        if len(completion.choices) > 0:
            for choice in completion.choices:
                content = choice.message.content
                res[i] = content
    json.dump(res,open(f'{file_name}_license.json','w'))


def calculate_lico(striking_sim,A_perm,A_copy,w1=1,w2=2,w3=4,total=4187):
    if A_perm == None:
        A_perm = 1
    if A_copy == None:
        A_copy = 1
    N = striking_sim / total

    lico = (w1 * (1 - N)  + w2 * A_perm + w3 * A_copy) / (w1 + w2 + w3)
    lico = round(lico,3)
    return lico  

def calculate_res(m):
    file_name = m.split('/')[-1]
    license_res = json.load(open(f'{file_name}_license.json'))
    answer = {}
    c=0
    permissive = 0
    pc = 0
    copyleft = 0
    cc=0
    for i in license_res:
        license_answered = find_license_by_keywords(license_res[i])
        if benchmark[i]["license_type"] == "Permissive":
            permissive+=1
        else:
            copyleft+=1
        answer[i]=(benchmark[i]["license"],benchmark[i]["license_type"],license_answered, 1 if license_answered == benchmark[i]["license"] else 0)
        if license_answered == benchmark[i]["license"]:
            c+=1
        if license_answered == benchmark[i]["license"] and benchmark[i]["license_type"] == "Permissive":
            pc+=1
        if license_answered == benchmark[i]["license"] and benchmark[i]["license_type"] != "Permissive":
            cc+=1
    json.dump(answer,open(f'{file_name}_license_answer.json','w'))
    
    if len(license_res) > 0:
        print("#striking_sim:",len(license_res),"({:.2f}%)".format(len(license_res)/len(benchmark)*100))
        print("#Acc:",c/len(license_res))
        print("#Permissive:",permissive)
        print("#Copyleft:",copyleft)
        if permissive > 0:
            print("#Permissive Acc:",pc/permissive)
        if copyleft > 0:
            print("#Copyleft Acc:",cc/copyleft)
        
        print("lico:",calculate_lico(len(license_res),pc/permissive if permissive > 0 else None, cc/copyleft if copyleft > 0 else None))
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt-4o", type=str
    )
    args = parser.parse_args()
    if args.model:
        comp_res = query(args.model)
        extracted_res = extrect_answer(comp_res,args.model)
        bleu,jaccard,edit_distance = calculate_sim(extracted_res,args.model)
        striking_sim = detect_striking_sim(extracted_res,bleu,jaccard,edit_distance,args.model)
        ask_for_license(striking_sim,comp_res,args.model)
        calculate_res(args.model)
#python3 compliance_test_completion.py --model gpt-4o
