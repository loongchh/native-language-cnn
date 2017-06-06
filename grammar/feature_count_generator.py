mode = "train"
maxtot = 2
d = {}
d["1"] = 0
d["2"] = 0
d["3"] = 0
d["4"] = 0
d["5"] = 0
d["6"] = 0
d["7"] = 0
d["8"] = 0
d["9"] = 0
d["10"] = 0
d["11"] = 0
d["12"] = 0
d["13"] = 0
d["14"] = 0
d["15"] = 0
d["16"] = 0
d["17"] = 0
d["18"] = 0
d["19"] = 0
d["20"] = 0
d["21"] = 0
d["22"] = 0
d["23"] = 0
d["24"] = 0
d["25"] = 0
d["26"] = 1
d["27"] = 0
d["28"] = 0
d["29"] = 0
d["30"] = 0
d["31"] = 0
d["32"] = 0
d["33"] = 0
d["34"] = 0
d["35"] = 0
d["36"] = 0
d["37"] = 0
d["38"] = 0

d["ARA"] = 0
d["CHI"] = 1
d["FRE"] = 2
d["GER"] = 3
d["HIN"] = 4
d["ITA"] = 5
d["JPN"] = 6
d["KOR"] = 7
d["SPA"] = 8
d["TEL"] = 9
d["TUR"] = 10
f = open(mode+'.tagged_data.txt', "r")
lines = f.readlines()
unvectors = []
nvectors = []
y = 0
maxlen = 0
totals = []
for line in lines:
    count = [0]*maxtot
    if line == "\n":
        unvectors.append([str(x) for x in count])
        nvectors.append([str(x) for x in ncount])
        totals.append(0.0)
        continue
    tags = [x.strip() for x in line.split(',')]
    for tag in tags:
        count[d[tag]] += 1
    total = len(tags)
    if total > maxlen:
        maxlen = total
    ncount = [float(x) / total for x in count]
    totals.append(float(total))
    unvectors.append([str(x) for x in count])
    nvectors.append([str(x) for x in ncount])
    y += 1
f.close()
for i in range(len(totals)):
    nvectors[i].append(str(totals[i]/maxlen))

g = open(mode+'_normalized.txt', "w+")
for nvector in nvectors:
    s = ",".join(nvector) + "\n"
    g.write(s)
g.close()
h = open(mode+'_unnormalized.txt', "w+")
for vector in unvectors:
    s = ",".join(vector) + "\n"
    h.write(s)
h.close()

res = {}
if mode == "train":
    x = open("labels_train.txt", "r")
    lines = x.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line in res:
            res[line] = [int(y) + int(z) for (y, z) in zip(res[line], unvectors[i])]           
        else:
            res[line] = unvectors[i]
    x.close()
    y = open("counts_train.txt", "w+")
    for k in res:
        line = [str(x) for x in res[k]]
        s = k + "\n" + ",".join(line) + "\n"
        y.write(s)
    y.close()
