import pickle
import preprocessing
from deepwalk import deepwalk
import os
import collections

opcode_idx_list = []

def vocBuild(blockIdxToTokens):
    global opcode_idx_list
    vocabulary = []
    reversed_dictionary = dict()
    count = [['UNK'], -1]
    index = 0
    for idx in blockIdxToTokens:
        for token in blockIdxToTokens[idx]:
            vocabulary.append(token)
            if token not in reversed_dictionary:
                reversed_dictionary[token] = index
                if token in preprocessing.opcode_list and index not in opcode_idx_list:
                    opcode_idx_list.append(index)
                index = index + 1
                
    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))
    count.extend(collections.Counter(vocabulary).most_common(1000 - 1))
    print('20 most common tokens: ', count[:20])
    del vocabulary
    return dictionary, reversed_dictionary

if __name__ == '__main__':
    biname = '/chroot'
    path = 'experiment_data/coreutils/binaries/'
    outputDir = 'output/'
    EDGELIST_FILE = outputDir + "edgelist"

    storage = []
    binaries = os.listdir(path)
    for i in range(0, len(binaries) - 1):
        for j in range(i + 1, len(binaries)):
            storage.append((path + binaries[i] + biname, path + binaries[j] + biname))

    index = 0
    if not os.path.isdir('data'):
        os.mkdir('data')
    for element in storage:
        blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts, _, _, bin1_name, bin2_name, toBeMergedBlocks = preprocessing.preprocessing(element[0], element[1], outputDir)
        dictionary, reversed_dictionary = vocBuild(blockIdxToTokens)
        walks = deepwalk.randomWalksGen(EDGELIST_FILE, blockIdxToTokens)
        p = os.getcwd()
        os.chdir('data')
        if not os.path.isdir(str(index)):
            os.mkdir(str(index))
        with open(str(index) + '/walks.pkl', 'wb') as file:
            pickle.dump(walks, file)
        with open(str(index) + '/dictionary.pkl', 'wb') as file:
            pickle.dump(dictionary, file)
        with open(str(index) + '/bb2token.pkl', 'wb') as file:
            pickle.dump(blockIdxToTokens, file)
        index += 1
        os.chdir(p)