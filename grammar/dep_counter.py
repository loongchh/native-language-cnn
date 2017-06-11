import operator
f = open("labels_train.txt", "r")
g = open("train_gram_features.txt", "r")
flines = f.readlines()
glines = g.readlines()

d={}
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

x = [[0]*37]*11
for i in range(len(flines)):
    fline = flines[i].strip()
    gline = glines[i].strip()
    gnums = gline.split(",")
    gfin = [int(y) for y in gnums]
    x[d[fline]] = map(operator.add, x[d[fline]], gfin)

f.close()
g.close()
h = open("train_gram_stats.txt", "w+")
for k in d.keys():
    h.write(k + "\n")
    arr = [str(y) for y in x[d[k]]]
    fin = ",".join(arr)
    h.write(fin + "\n")
h.close()
