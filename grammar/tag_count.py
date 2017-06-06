f = open('train.tagged_data.txt', "r")
lines = f.readlines()
count = [0]*38
for line in lines:
    
    if line == "\n":
        continue
    tags = [x.strip() for x in line.split(',')]
    for tag in tags:
        count[int(tag)-1] += 1
f.close()
print count
