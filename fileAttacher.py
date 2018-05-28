import numpy as np
from sets import Set

def main():
    smallset = 0
    total = 0
    labeled = 0
    unlabeled = 0

    if smallset:
        f = open("data_training_small", "r")
    else:
        f = open("data_training", "r")

    mydict = {}
    for l in f:
        total = total + 1
        #a = l.strip("\r").strip("\n").split("\t")
        a = (''.join(unicode(l, 'utf-8').splitlines())).split("\t")
        mydict[a[0]] = ("Neutral", a[1])
    f.close()

    if smallset:
        f = open("output_training_small", "r")
    else:
        f = open("output_training", "r")

    for l in f:
        a = l.split("\t")
        if(a[0] in mydict):
            mydict[a[0]] = (a[1], mydict[a[0]][1])
            labeled = labeled + 1
        else:
            unlabeled = unlabeled + 1
    f.close()

    if smallset:
        f = open("TESTE", "w")
    else:
        f = open("TESTE_FULL", "w")

    for x in mydict:
        #f.write(mydict[x][0]+"\t"+mydict[x][1]+"\n")
        f.write((mydict[x][0]+"\t"+mydict[x][1]+"\n").encode('utf-8'))
    f.close()

    print "TOTAL TWEETS = ",total,"\nLABELED = ",labeled,"\nUNLABELED = ",unlabeled

    #mydict = {a[0]:("Neutral", a[1]) for a in [l.split('\t') for l in f]}


if __name__ == "__main__":
    main()
