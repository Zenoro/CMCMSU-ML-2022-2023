import re

l='fr3abc4'
# for i in range (len(l))
k = [c if not re.search(r'[^a-zA-Z]',l[c]) else -1 for c in range(len(l))]
k=k.split(-1)
k=sorted(k,key=len)
print(len(k[0])
