def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

import os
from nltk.parse import stanford
mode = "dev"
os.environ['STANFORD_PARSER'] = "stanford-parser-full-2016-10-31/stanford-parser.jar"
os.environ['STANFORD_MODELS'] = "stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar"
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jre1.8.0_102/bin/"
os.environ['STANFORD_CORENLP'] = "stanford-parser-full-2016-10-31/stanford-english-corenlp-2016-10-31-models.jar"
parser = stanford.StanfordDependencyParser("stanford-parser-full-2016-10-31/englishPCFG.ser.gz")

docs = []
dmap = {}
fmap = {}
count = 0
dcount = 0
for f in os.listdir("../data/speech_transcriptions/"+ mode + "/original"):
    with open("../data/speech_transcriptions/" + mode + "/original/"+f, "r") as g:
        doc = g.read()
        words = doc.split(" ")
        for i in range(0, len(words)/50 + 1):
            chunk = " ".join(words[50*i:50*i+50])
            docs.append(strip_non_ascii(chunk))
            dmap[count] = dcount
            fmap[count] = f
            count += 1
        dcount +=1
        if(dcount%1000 == 999):
            print count

sentences = parser.raw_parse_sents(docs)
x = [y for y in sentences]
print(len(x))
d = [u'nsubj', u'det', u'acl:relcl', u'dep', u'advmod', u'cc', u'conj', u'cop', u'compound', u'appos', u'nmod', u'case', u'dobj', u'mark', u'aux', u'amod', u'nmod:npmod', u'nummod', u'xcomp', u'discourse', u'advcl', u'nmod:poss', u'acl', u'nsubjpass', u'auxpass', u'ccomp', u'mwe', u'parataxis', u'neg', u'csubj', u'det:predet', u'expl', u'compound:prt', u'iobj', u'nmod:tmod', u'cc:preconj', u'csubjpass']
g = {}
w = {}
nmap = {}
dgs = []
for i in range(len(x)):
    z = [a for a in x[i]]
    dgs.append(z)
    c = list(z[0].triples())
    words = [a['word'] for a in z[0].nodes.values()]
    fwords = []
    for word in words:
        if type(u'I') == type(word):
            fwords.append(word)
    finstring = " ".join(fwords)
    if dmap[i] in g:
        l = g[dmap[i]]
        q = w[dmap[i]]
    else:
        g[dmap[i]] = {}
        w[dmap[i]] = ""
        nmap[dmap[i]] = fmap[i]
        l = g[dmap[i]]
        q = w[dmap[i]]
    w[dmap[i]] += " " + finstring
    for j in c:
        if j[1] not in d:
            d.append(j[1])
        if j[1] not in l:
            l[j[1]] = 1
        else:
            l[j[1]] += 1

h = open(mode + "_dep.txt", "w+")
k = sorted(g.keys())
print(len(k))
for key in k:
    out = []
    for val in d:
        if val in g[key]:
            out.append(str(g[key][val]))
        else:
            out.append("0")
    h.write(str(key) + "\n" + w[key] + "\n" + ",".join(out) + "\n")
h.close()
