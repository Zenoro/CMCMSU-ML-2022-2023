def check(x: str, file: str):
    counter = 0
    with open(file, "w") as f:
        text = x.lower().rstrip().split()
        sorted_x = sorted(list(set(x.lower().split()))) 
        for i in range(0,len(sorted_x)):
            for j in range (0,len (text)):
                if sorted_x[i] == text[j]:
                    counter += 1
            f.write(sorted_x[i]+" "+str(counter)+"\n") 
            counter = 0
