import tqdm
import json
from collections import defaultdict
import re
import ast
def parse_func_regex(content, repo_name="", repo_path=""):
    # 处理文件头部的标签
    file_header_added = ""
    if "<reponame>" in content:
        content = content.replace("<reponame>"+repo_name, "")
        file_header_added += "<reponame>"+repo_name
    if "<filename>" in content:
        content = content.replace("<filename>"+repo_path, "")
        file_header_added += "<filename>"+repo_path
    if "<gh_stars>" in content:
        stars = re.findall(r'<gh_stars>(\d+-\d+|\d+)', content)
        file_header_added += "<gh_stars>"+stars[0]
        content = content.replace("<gh_stars>"+stars[0], "")

    content = content.replace('\r\n', '\n')

    # 提取函数定义和docstring
    pattern =r"(def .+?:\s+?)([ \t]*\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')?"
    matches = re.findall(pattern, content, re.DOTALL)

    all_functions = []
    for i, match in enumerate(matches):
        # 提取函数名和签名
        function_signature = match[0].strip()
        #function_name = function_signature.split(' ')[1].split('(')[0]
        # 提取docstring
        docstring = match[1] if match[1] else ""
        # 提取函数体
        function_body_start = content.index(match[0]) + len(match[0]) + len(match[1])
        function_body_end = content.index("\ndef ", function_body_start) if "\ndef " in content[function_body_start:] else len(content)
        function_body = content[function_body_start:function_body_end]

        # 提取文件头部
        if i == 0:
            file_header = content[:content.index(match[0])]
        # else:
        #     file_header = ""

        # 添加到列表
        all_functions.append((file_header_added + file_header + "\n" + function_signature +"\n" + docstring, file_header,function_signature,docstring, function_body, 1 if docstring else 0))

    return all_functions



def paser_func(content,repo_name="",repo_path=""):
    
    #starcoder数据集特有的，随机在文件头部添加了<reponame>和<filename>等标签
    file_header_added = ""
    if "<reponame>" in content:
        content = content.replace("<reponame>"+repo_name, "")
        file_header_added += "<reponame>"+repo_name
    if "<filename>" in content:
        content = content.replace("<filename>"+repo_path, "")
        file_header_added += "<filename>"+repo_path
    if "<gh_stars>" in content:
        stars = re.findall(r'<gh_stars>(\d+-\d+|\d+)', content)
        file_header_added += "<gh_stars>"+stars[0]
        content = content.replace("<gh_stars>"+stars[0], "")
        
    content = content.replace('\r\n', '\n')
    try:
        tree = ast.parse(content)
    except:
        return parse_func_regex(content,repo_name,repo_path)
    lines = content.split('\n')

    # 查找第一个函数开始的行，然后将这之前的所有内容作为文件头部
    file_header = None
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            file_header = '\n'.join(lines[:item.lineno-1])
            break

    all_functions = []

    def get_end_lineno(node):
        """Recursively find the last line number of a node."""
        max_lineno = getattr(node, 'lineno', -1)
        for child in ast.iter_child_nodes(node):
            max_lineno = max(max_lineno, get_end_lineno(child))
        return max_lineno

    def get_docstring_end_lineno(node, lines):
        """Get the end line number of the docstring."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            docstring_lines = node.body[0].value.s.split('\n')
            return node.body[0].lineno + len(docstring_lines) - 1
        else:
            return node.lineno

    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):  # 如果是函数定义
            func_begin = item.lineno  # 获取函数开始的行号
            func_end = get_end_lineno(item)  # 获取函数结束的行号

            docstring = ast.get_docstring(item)  # 获取函数的docstring
            if docstring:  # 如果有docstring
                has_docstring = 1  # 设置 has_docstring 为1
                docstring_end = get_docstring_end_lineno(item, lines)
                function_signature = '\n'.join(lines[func_begin-1:func_begin])
                function_docstring = '\n'.join(lines[func_begin:docstring_end])
            else:  # 如果没有docstring
                has_docstring = 0  # 设置 has_docstring 为0
                function_signature = '\n'.join(lines[func_begin-1:func_begin])
                function_docstring = ""
                docstring_end = func_begin

            function_body = '\n'.join(lines[docstring_end:func_end])  # 获取函数体

            all_functions.append((file_header_added+file_header+"\n"+function_signature+"\n"+function_docstring, file_header, function_signature,function_docstring, function_body, has_docstring))  # 将函数添加到列表中

    return all_functions

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


    