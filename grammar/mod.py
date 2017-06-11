f = open("temprac.txt", "r")
h = open("tempdevf.txt", "w+")
lines = f.readlines()
prev = False
prev2 = False
prev2Line = ""
prevLine = ""
print(len(lines))
for line in lines:
    words = line.split(" ")
    if(words[0].endswith(".txt")):
        if(not(prev)):
            h.write(prevLine)
        prev = True
    else:
        h.write(prevLine)
        prev = False
    prevLine = line
h.close()
f.close()
