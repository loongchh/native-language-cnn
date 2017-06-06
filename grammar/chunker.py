def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

import os


mode = "dev"

h = open("joinedfiledev.txt", "w+")
count = 0
for f in os.listdir("../data/speech_transcriptions/"+ mode + "/original"):
    with open("../data/speech_transcriptions/" + mode + "/original/"+f, "r") as g:
        doc = g.read()
        words = doc.split(" ")
        for i in range(0, len(words)/50 + 1):
            chunk = " ".join(words[50*i:50*i+50])
            h.write(f + " " + str(count) + "\n" + chunk + "\n")
            count += 1
h.close()
