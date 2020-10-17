import pandas as pd
import nltk
from os import remove
import os
import pymongo
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.neighbors import NearestNeighbors


def collect_data():
    df = pd.read_csv("songdata.csv").dropna()
    df = df.drop(columns=["link"])
    return df.sample(1000, random_state=3)


def to_txt(df: pd.DataFrame):
    file_count = 0
    for _index, row in df.iterrows():
        path = "data/{}.txt".format(file_count)
        try:
            with open(path, "w") as f:
                f.write(row["artist"])
                f.write("\n")
                f.write(row["song"])
                f.write("\n")
                f.write("\n")
                f.write(row["text"])
                file_count += 1
        except:
            remove(path)


def pre_process(df: pd.DataFrame):
    stemmer = nltk.PorterStemmer()
    clean_data = []
    symbols = "',!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    for _index, row in df.iterrows():
        data = row["artist"] + ' / ' + row["song"] + ' / ' + row["text"]

        clean_data.append(
            [stemmer.stem(word.lower()) for word in nltk.word_tokenize(data) if
             word not in symbols and word.lower()])
        # not in stopwords
    return clean_data


def pre_process_query(raw_query):
    stemmer = nltk.PorterStemmer()
    symbols = "',!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    clean_query = (
        [stemmer.stem(word.lower()) for word in nltk.word_tokenize(raw_query)
         if word not in symbols and word.lower()])
    return clean_query


def get_from_db():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["atdb"]
    collection = db["lyrics"]
    lyrics = collection.find()
    data = []
    indices = []
    endpoint = [0]
    for song in lyrics:
        for (key, value) in zip(song["text"].keys(), song["text"].values()):
            data.append(value)
            indices.append(key)
        endpoint.append(endpoint[-1] + len(song["text"].values()))
    tf = csr_matrix((data, indices, endpoint))
    return tf


# Data Transformation
def nothing(doc):
    return doc


def create_db():
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=nothing,
        preprocessor=nothing,
        token_pattern=None)

    to_txt(collect_data())
    clean_data = pre_process(collect_data())
    tfidf.fit(clean_data)
    X = tfidf.transform(clean_data)
    pickle.dump(tfidf, open("tfidf.pickle", "wb"))

    my_client = pymongo.MongoClient("mongodb://localhost:27017/")
    my_db = my_client["atdb"]
    my_col = my_db["lyrics"]

    for i in range(0, len(clean_data)):
        post = dict()
        post["_id"] = str(i)
        tf_dict = dict()

        for j in range(len(X[i].data)):
            tf_dict[str(X[i].indices[j])] = X[i].data[j]

        post["text"] = tf_dict
        post["link"] = os.path.abspath("data/{}.txt".format(i))
        my_col.insert_one(post)


def nearest_k(query, k, dist):
    open_tf = pickle.load(open("tfidf.pickle", "rb"))
    clean_query = pre_process_query(query)
    tf_query = open_tf.transform([clean_query])
    from_db = get_from_db()

    if dist == "jaccard" or dist == "chebyshev":
        from_db = from_db.todense()
        tf_query = tf_query.todense()

    nbrs = NearestNeighbors(n_neighbors=k, metric=dist).fit(from_db)
    distances, indices = nbrs.kneighbors(tf_query)  # return_distance=False

    return distances[0], indices[0]
