

wol = open('data_shuffled.txt').readlines()
rtfeat = {}
rtword = {}
for w in range(len(wol)):
    wl = wol[w][:-1].split(' ')
    rtfeat[wl[1]] = wl[-1]
    rtword[wl[1]] = wl[0]

dct = open('wol-gofa').readlines()
wgdict = {}
for w in range(len(dct)):
    wl = dct[w][:-1].split(' ')
    wgdict[wl[0]] = wl[1]

gof = open('r-r.txt').readlines()
new = open("gof.txt", mode='w')
for g in range(len(gof)):
    gs = gof[g][:-1].split(' ')
    groot = gs[1]
    wroot = gs[0]
    if wroot in rtfeat and rtword[wroot] in wgdict:
        wword = rtword[wroot]
        gword = wgdict[rtword[wroot]]
        feat = rtfeat[wroot]
        line = "{0} {1} {2}\n".format(groot, gword, feat)
        new.write(line)
new.close()

