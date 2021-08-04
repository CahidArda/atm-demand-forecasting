import pickle

def load_pickle(path):
    file = open(path, 'rb')
    d = pickle.load(file)
    file.close()
    return d

d = load_pickle('pickles/Day_of_the_Week_Index_cluster_pickle')
print(d)