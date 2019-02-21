word_word = open('data_proc/word_word.txt').readlines()
root_root = open('data_proc/root_root.txt').readlines()
wol_feat = open('data_proc/data_shuffled.txt').readlines()
wword2feat = {}
wrrot2feat = {}
final_output = {}
for line in wol_feat:
    line = line[:-1].split(' ')
    wword2feat[line[0]] = line[2]
    wrrot2feat[line[1]] = line[2]

file = open('final.txt', mode='w', encoding='utf-8')
for i in range(len(word_word)):
    wline  = word_word[i]
    wline = wline[:-1].split(' ')
    wword = wline[0]
    gword = wline[1]

    rline = root_root[i]
    rline = rline[:-1].split(' ')
    wroot = rline[0]
    groot = rline[1]

    if wword in wword2feat:
        wfeat = wword2feat[wword]
        line = "{0} {1} {2}\n".format(gword, groot, wfeat)
        file.write(line)


    
