import re

def find_shortest(l):
    s=re.findall('[a-zA-Z]+', l)
    return len(min(s, key=len)) if s else 0