import pickle

PATH = "article-dataset/params/bach.pkl"

with open(PATH, 'rb') as f1:
    OL = pickle.load(f1)
    print(OL)