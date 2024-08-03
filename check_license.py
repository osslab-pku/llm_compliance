
import json
import re



init_keyword=json.load(open("keywords.json","r"))
keyword={}
for key in init_keyword:
    keyword[key.lower()]=init_keyword[key]

def have_keywords(dirty,clean,ver=False):
    if clean in ["gpl-2.0","gpl-3.0","lgpl-2.1","lgpl-3.0","agpl-3.0"]:
        clean=clean+"-only"
    dirty=dirty.lower()
    if clean not in keyword:
        return False
    req=keyword[clean]
    def check(name):
        if " " in name:
            return name in dirty
        else:
            return name in dirty.split()
    if "name" in req:
        if len(list(filter(check,req["name"]))) == 0:
            return False
    if "must" in req:
        if len(list(filter(lambda e:e in dirty,req["must"]))) == 0:
            return False
    if "no" in  req:
        if len(list(filter(lambda e:e in dirty,req["no"]))) != 0:
            return False
    if ver and "version" in req:
        if len(list(filter(lambda e: re.findall(r"(?<=v|-|\s)"+e+r"(?!\d)",dirty),req["version"]))) == 0:
                           
            return False
    return True

def find_license_by_keywords(dirty,version=True):
    for clean in keyword:
        if have_keywords(dirty,clean,version):
            return clean
    return None

def detect_dual_license(dirty,version=True):
    license =[]
    for clean in keyword:
        if have_keywords(dirty,clean,version):
            license.append(clean)
    if len(license) > 1:
        return True
    return False



    
        
