def normalize(filename, outputname):
    f = open(filename, "r")
    h = open(outputname, "w+")
    lines = f.readlines()
    for line in lines:
        splitter = line.split(",")
        vals = [int(val) for val in splitter]
        s = float(sum(vals))
        if s == 0.0:
            nvals = ["0.0" for val in vals]
        else:
            nvals = [str(val/s) for val in vals]
        h.write(",".join(nvals) + "\n")
    h.close()
    f.close()
