from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords


def tf_idf(cleaned_propositions, MAX_DF, MIN_DF, NO_OF_CLUSTERS, NO_OF_MEMBERS):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=100,
        max_df=MAX_DF,
        min_df=MIN_DF,
        ngram_range=(1,3), # (tuple) include words that are either 1 token, 2 tokens (bigram) or 3 tokens (trigram)
        stop_words=stopwords.words("french")
    )

    vectors = vectorizer.fit_transform(cleaned_propositions)

    feature_names = vectorizer.get_feature_names_out()

    dense = vectors.todense()
    denselist = dense.tolist()

    all_keywords = []

    for proposition in denselist:
        x = 0
        keywords = []
        for word in proposition:
            if word > 0:
                keywords.append(feature_names[x])
            x = x+1
        all_keywords.append(keywords)
    
    true_k = NO_OF_CLUSTERS # number of clusters/topics we want

    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

    model.fit(vectors)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    result = []
    for i in range(true_k):
        cluster=[]
        for ind in order_centroids[i, :NO_OF_MEMBERS]:
            cluster.append(terms[ind])
        result.append(cluster)
    
    return result