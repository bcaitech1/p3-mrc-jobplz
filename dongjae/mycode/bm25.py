class BM25(object):
    def __init__(self, tokenizer=tokenize, ngram_range=(1, 2), b=0.75, k1=1.5):
        # 만약 tfidfvectorizer가 있으면 불러와서 저장, fit함수 저장
        tfidfv_path = '/opt/ml/input/data/data/tfidv.bin'
        if os.path.isfile(tfidfv_path):
            with open(tfidfv_path, "rb") as file:
                self.vectorizer = pickle.load(file)
            self.is_fit = True
            print('load the tfidfv')
        else :
            self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False,
                                          tokenizer=tokenize, ngram_range=(1, 2))
            self.is_fit = False
        self.b = b
        self.k1 = k1

    def fit_transform(self, context):
        if not self.is_fit :
            self.vectorizer.fit(context)
            self.is_fit = True 
        y = super(TfidfVectorizer, self.vectorizer).transform(context)
        self.avdl = y.sum(1).mean()

        b, k1, avdl = self.b, self.k1, self.avdl

        len_y = y.sum(1).A1

        y = y.tocsc()
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        numer = y * (k1 + 1)
        p_embedding = (numer / denom)
        return csr_matrix(p_embedding, dtype = np.float16)

    def fit(self, X):
        """ Fit IDF to documents X """
        if not self.is_fit :
            self.vectorizer.fit(X)
            self.is_fit = True 
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, X):
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        assert sparse.isspmatrix_csr(X)
        idf = self.vectorizer._tfidf.transform(X, copy=False)
        #idf.todense()
        #idf.data -= 1
        #idf.tocsc()
        return idf