def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

from operator import add
import os
mode = "dev"


h = open("Feature Pair Dev.txt", "r")
count = 0
files = os.listdir("../data/speech_transcriptions/" + mode + "/original")
od = {}
o = open("outputfiledev.txt", "r")
lines = o.readlines()
i = 0
while i < len(lines):
    line1 = lines[i]
    line2 = lines[i+1]
    vals = line2.split(",")
    od[line1] = [int(val) for val in vals]
    i+=2
print(len(od.keys()))
hlines = h.readlines()
hd = {}
for line in hlines:
    pair = line.split(" ")
    p1 = pair[0]
    p2 = pair[1]
    if p1 in hd:
        hd[p1] = map(add, hd[p1], od[p2])
    else:
        hd[p1] = od[p2]
o.close()
h.close()
last = open("dev_gram_features.txt", "w+")
for f in files:
    if f in hd:
        n = [str(val) for val in hd[f]]
        last.write(",".join(n) + "\n")
    else:
        last.write("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
last.close()
