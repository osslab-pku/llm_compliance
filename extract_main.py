from empirical_study_for_striking_sim.data_processing.accessed import paser_func
import tqdm
import json
from collections import defaultdict
import re
def remove_docstring(code):
    pattern = r'^\s*(""".*?"""|\'\'\'.*?\'\'\')(\n|\s)*'  
    docstrings = re.findall(pattern, code, flags=re.DOTALL)
    if docstrings:
        docstring = docstrings[0][0]
        code = code[code.index(docstring)+len(docstring):]
    return code

def end_at_return(code):
    extra_sigs = re.findall(r"\ndef",code)
    if extra_sigs:
        code = code[:code.index(extra_sigs[-1])]
    return code
def get_signature(s):
    #pattern = r"(def\s+\.+\s*\()"
    pattern = r"(\ndef\s+.+\s*\()"
    signature = re.findall(pattern, s)
    if signature:
        return signature[0]
    else:
        return None
    
def del_name(code):
    if '__name__ == "__main__":' in code:
        next_line = code.index('__name__ == "__main__":')
        code = code[:next_line].strip()
    elif "__name__ == '__main__':" in code:
        next_line = code.index("__name__ == '__main__':")
        code = code[:next_line].strip()
    
    if "# Example usage" in code:
        # print(completion)
        next_line = code.index('# Example usage')
        code = code[:next_line].strip()
    return code
def extract(code,signature):
    codes = re.findall(r'```python\s*(.*?)```', code, re.DOTALL)
    if len(codes) == 0:
        codes = re.findall(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
        if len(codes) == 0:    
            codes = re.findall(r'```(?:python)?\s*(.*?)```', code+"```", re.DOTALL)
    if codes:
        code = codes[0]
    if signature:
        try:
            code = code[code.index(signature)+len(signature):]        
            body = code[code.find(':\n')+2:]
            body = remove_docstring(body)
            body = end_at_return(body)
            body = del_name(body)
            return body 
        except:
            pass
    signature = get_signature(code)
    if signature: 
        signature = signature.split('(')[0]    
        try:
            code = code[code.index(signature)+len(signature):]        
            body = code[code.find(':\n')+2:]
            body = remove_docstring(body)
            body = end_at_return(body)  
            body = del_name(body)
            return body 
        except:
            code = remove_docstring(code)
            body = end_at_return(code)
            body = del_name(body)
    
    code = remove_docstring(code)
    body = end_at_return(code)
    body = del_name(body)

    return body



if __name__ == '__main__':
    res = json.load(open(f'poro/unseen10k_eval_t_0.0_p_1.0.json'))
    res_extracted = defaultdict(list)
    unseen_samples = json.load(open(f'unseen_samples_10k_eval.json'))
    for i in tqdm.tqdm(res):
        signature = paser_func(unseen_samples[i]["header"]+"\n"+unseen_samples[i]["body"])[0][2]
        signature = signature.split('(')[0]
        code = res[i][0]
        code = extract(code,signature)
        res_extracted[i].append(code.strip("\n"))
    json.dump(res_extracted,open(f'poro/unseen10k_eval_extracted_t_0.0_p_1.0.json','w'))


    