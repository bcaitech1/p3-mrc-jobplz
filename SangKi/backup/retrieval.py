import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
from utils_qa import BM25

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager
import kss
import pickle

stopword = "아 휴 아이구 아이쿠 아이고 어 나 우리 저희 따라 의해 을 를 에 의 가 으로 로 에게 뿐이다 의거하여 근거하여 입각하여 기준으로 예하면 예를 들면 예를 들자면 저 소인 소생 저희 지말고 하지마 하지마라 다른 물론 또한 그리고 비길수 없다 해서는 안된다 뿐만 아니라 만이 아니다 만은 아니다 막론하고 관계없이 그치지 않다 그러나 그런데 하지만 든간에 논하지 않다 따지지 않다 설사 비록 더라도 아니면 만 못하다 하는 편이 낫다 불문하고 향하여 향해서 향하다 쪽으로 틈타 이용하여 타다 오르다 제외하고 이 외에 이 밖에 하여야 비로소 한다면 몰라도 외에도 이곳 여기 부터 기점으로 따라서 할 생각이다 하려고하다 이리하여 그리하여 그렇게 함으로써 하지만 일때 할때 앞에서 중에서 보는데서 으로써 로써 까지 해야한다 일것이다 반드시 할줄알다 할수있다 할수있어 임에 틀림없다 한다면 등 등등 제 겨우 단지 다만 할뿐 딩동 댕그 대해서 대하여 대하면 훨씬 얼마나 얼마만큼 얼마큼 남짓 여 얼마간 약간 다소 좀 조금 다수 몇 얼마 지만 하물며 또한 그러나 그렇지만 하지만 이외에도 대해 말하자면 뿐이다 다음에 반대로 반대로 말하자면 이와 반대로 바꾸어서 말하면 바꾸어서 한다면 만약 그렇지않으면 까악 툭 딱 삐걱거리다 보드득 비걱거리다 꽈당 응당 해야한다 에 가서 각 각각 여러분 각종 각자 제각기 하도록하다 와 과 그러므로 그래서 고로 한 까닭에 하기 때문에 거니와 이지만 대하여 관하여 관한 과연 실로 아니나다를가 생각한대로 진짜로 한적이있다 하곤하였다 하 하하 허허 아하 거바 와 오 왜 어째서 무엇때문에 어찌 하겠는가 무슨 어디 어느곳 더군다나 하물며 더욱이는 어느때 언제 야 이봐 어이 여보시오 흐흐 흥 휴 헉헉 헐떡헐떡 영차 여차 어기여차 끙끙 아야 앗 아야 콸콸 졸졸 좍좍 뚝뚝 주룩주룩 솨 우르르 그래도 또 그리고 바꾸어말하면 바꾸어말하자면 혹은 혹시 답다 및 그에 따르는 때가 되어 즉 지든지 설령 가령 하더라도 할지라도 일지라도 지든지 몇 거의 하마터면 인젠 이젠 된바에야 된이상 만큼 어찌됏든 그위에 게다가 점에서 보아 비추어 보아 고려하면 하게될것이다 일것이다 비교적 좀 보다더 비하면 시키다 하게하다 할만하다 의해서 연이서 이어서 잇따라 뒤따라 뒤이어 결국 의지하여 기대여 통하여 자마자 더욱더 불구하고 얼마든지 마음대로 주저하지 않고 곧 즉시 바로 당장 하자마자 밖에 안된다 하면된다 그래 그렇지 요컨대 다시 말하자면 바꿔 말하면 즉 구체적으로 말하자면 시작하여 시초에 이상 허 헉 허걱 바와같이 해도좋다 해도된다 게다가 더구나 하물며 와르르 팍 퍽 펄렁 동안 이래 하고있었다 이었다 에서 로부터 까지 예하면 했어요 해요 함께 같이 더불어 마저 마저도 양자 모두 습니다 가까스로 하려고하다 즈음하여 다른 다른 방면으로 해봐요 습니까 했어요 말할것도 없고 무릎쓰고 개의치않고 하는것만 못하다 하는것이 낫다 매 매번 들 모 어느것 어느 로써 갖고말하자면 어디 어느쪽 어느것 어느해 어느 년도 라 해도 언젠가 어떤것 어느것 저기 저쪽 저것 그때 그럼 그러면 요만한걸 그래 그때 저것만큼 그저 이르기까지 할 줄 안다 할 힘이 있다 너 너희 당신 어찌 설마 차라리 할지언정 할지라도 할망정 할지언정 구토하다 게우다 토하다 메쓰겁다 옆사람 퉤 쳇 의거하여 근거하여 의해 따라 힘입어 그 다음 버금 두번째로 기타 첫번째로 나머지는 그중에서 견지에서 형식으로 쓰여 입장에서 위해서 단지 의해되다 하도록시키다 뿐만아니라 반대로 전후 전자 앞의것 잠시 잠깐 하면서 그렇지만 다음에 그러한즉 그런즉 남들 아무거나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 어떻게 만약 만일 위에서 서술한바와같이 인 듯하다 하지 않는다면 만약에 무엇 무슨 어느 어떤 아래윗 조차 한데 그럼에도 불구하고 여전히 심지어 까지도 조차도 하지 않도록 않기 위하여 때 시각 무렵 시간 동안 어때 어떠한 하여금 네 예 우선 누구 누가 알겠는가 아무도 줄은모른다 줄은 몰랏다 하는 김에 겸사겸사 하는바 그런 까닭에 한 이유는 그러니 그러니까 때문에 그 너희 그들 너희들 타인 것 것들 너 위하여 공동으로 동시에 하기 위하여 어찌하여 무엇때문에 붕붕 윙윙 나 우리 엉엉 휘익 윙윙 오호 아하 어쨋든 만 못하다 하기보다는 차라리 하는 편이 낫다 흐흐 놀라다 상대적으로 말하자면 마치 아니라면 쉿 그렇지 않으면 그렇지 않다면 안 그러면 아니었다면 하든지 아니면 이라면 좋아 알았어 하는것도 그만이다 어쩔수 없다 하나 일 일반적으로 일단 한켠으로는 오자마자 이렇게되면 이와같다면 전부 한마디 한항목 근거로 하기에 아울러 하지 않도록 않기 위해서 이르기까지 이 되다 로 인하여 까닭으로 이유만으로 이로 인하여 그래서 이 때문에 그러므로 그런 까닭에 알 수 있다 결론을 낼 수 있다 으로 인하여 있다 어떤것 관계가 있다 관련이 있다 연관되다 어떤것들 에 대해 이리하여 그리하여 여부 하기보다는 하느니 하면 할수록 운운 이러이러하다 하구나 하도다 다시말하면 다음으로 에 있다 에 달려 있다 우리 우리들 오히려 하기는한데 어떻게 어떻해 어찌됏어 어때 어째서 본대로 자 이 이쪽 여기 이것 이번 이렇게말하자면 이런 이러한 이와 같은 요만큼 요만한 것 얼마 안 되는 것 이만큼 이 정도의 이렇게 많은 것 이와 같다 이때 이렇구나 것과 같이 끼익 삐걱 따위 와 같은 사람들 부류의 사람들 왜냐하면 중의하나 오직 오로지 에 한하다 하기만 하면 도착하다 까지 미치다 도달하다 정도에 이르다 할 지경이다 결과에 이르다 관해서는 여러분 하고 있다 한 후 혼자 자기 자기집 자신 우에 종합한것과같이 총적으로 보면 총적으로 말하면 총적으로 대로 하다 으로서 참 그만이다 할 따름이다 쿵 탕탕 쾅쾅 둥둥 봐 봐라 아이야 아니 와아 응 아이 참나 년 월 일 령 영 일 이 삼 사 오 육 륙 칠 팔 구 이천육 이천칠 이천팔 이천구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 령 영 이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤 내 내 경우 명 생각 시간 그녀 다시 이런 앞 보이 번 나 다른 어떻 여자 개 전 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 놓".split()

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


# 개행 치환
def remove_newlines(example) :
    new_one = re.sub(r'\n', '', example)

    return new_one

# 더블 스페이스 제거
def remove_double_space(example) :
    new_one = ' '.join(example.split())

    return new_one

trash_ids = [973, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 5527,
             9079, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087, 9088, 28989, 29028,
              31111, 37157]


class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="/opt/ml/code/dongjae/mycode/data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

    # 기본
        # self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values() if not v['document_id'] in trash_ids])) # set 은 매번 순서가 바뀌므로
    # 위키피디아 text 전처리 -> 1. 개행제거, 2. 띄어쓰기 하나만
        # self.contexts = list(dict.fromkeys([remove_double_space(remove_newlines(v['text'])) for v in wiki.values()]))
    # 위키피디아 불용어 제거
        # self.temp_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        # self.contexts = []
        
        # for temp_context in self.temp_contexts :
        #     temp_result = []
        #     for word in temp_context.split() :
        #         if word not in stopword :
        #             temp_result.append(word)
        #     result = ' '.join(temp_result)
        #     self.contexts.append(result)

        # for temp_context in self.temp_contexts :
        #     temp_result = []
        #     for word in temp_context :
        #         if (word not in stopword) and (word not in ['_'+sw for sw in stopword]) :
        #             temp_result.append(word)
        #     result = ''
        #     for word in temp_result :
        #         if word[0] == '_' :
        #             result += ' ' + word[1:]
        #         else :
        #             result += word
        #     self.contexts.append(result)
    # 위키피디아 kss로 잘라서
        wiki1024_name = 'wiki_1024_kss.pickle'
        wiki784_name = 'wiki_784_kss.pickle'
        wiki1280_name = 'wiki_1280_kss.pickle'
        wiki1500_name = 'wiki_1500_kss.pickle'

        if os.path.isfile(os.path.join(self.data_path, wiki1280_name)) :
            with open(os.path.join(self.data_path, wiki1280_name), 'rb') as file :
                self.contexts = pickle.load(file)
        else :
            new_context = [v['text'] for v in wiki.values() if v['document_id'] not in trash_ids]
            self.contexts = []
            for current in new_context :
                # print(current)
                try :
                    te = kss.split_chunks(current, max_length= 1280, overlap = True)
                    for v in te :
                        self.contexts.append(v.text)
                except :
                    self.contexts.append(current)
                    continue
            # 위키피디아 피클 저장
            with open(os.path.join(data_path, wiki1280_name), "wb") as file:
                pickle.dump(self.contexts, file)

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))



        # Transform by vectorizer
    # 기존 방법 - tfidf
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            # max_features=50000,
        )
    # BM25 - 용량이 터진다;
        # self.tfidfv = BM25(
        #         tokenizer=tokenize_fn, #형태소 자르기
        #         ngram_range=(1, 2),
        #         k1= 1.0
        # )
        # should run get_sparse_embedding() or build_faiss() first.
        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self):
        # Pickle save.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
    # 기본
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
    
    # # 변경
    #         print("Build passage embedding")
    #         self.tfidfv.fit(self.contexts)
    #         self.p_embedding = self.tfidf2bm25()
    #         print(self.p_embedding.shape)
    #         with open(emd_path, "wb") as file :
    #             pickle.dump(self.p_embedding, file)
    #         with open(tfidfv_path, "wb") as file:
    #             pickle.dump(self.tfidfv, file)
    #         print("Embedding pickle saved.")

    def tfidf2bm25(self,tfidf_path='/opt/ml/input/data/data/sparse_embedding.bin'):
        with open(tfidf_path,"rb") as file:
            csr = pickle.load(file)
        csr = csr.astype('float64')
        avdl = csr.sum(1).mean() * 0.8
        len_csr = csr.sum(1).A1
        k1, b, delta = 2.0, 0.75, 1
        for row, (B, E) in enumerate(zip(csr.indptr, csr.indptr[1:])):
            for idx in range(B, E):
                denom = csr.data[idx] + (k1 * (1.0 - b + b * len_csr[row] / avdl))
                numer = csr.data[idx] * (k1 + 1.0)
                csr.data[idx] = numer / denom + delta
    
        return csr

    def build_faiss(self):
        # FAISS build
        num_clusters = 16
        niter = 5

        # 1. Clustering
        p_emb = self.p_embedding.toarray().astype(np.float32)
        emb_dim = p_emb.shape[-1]
        index_flat = faiss.IndexFlatL2(emb_dim)

        clus = faiss.Clustering(emb_dim, num_clusters)
        clus.verbose = True
        clus.niter = niter
        clus.train(p_emb, index_flat)

        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(num_clusters, emb_dim)

        quantizer = faiss.IndexFlatL2(emb_dim)
        quantizer.add(centroids)

        # 2. SQ8 + IVF indexer (IndexIVFScalarQuantizer)
        self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)

    def retrieve(self, query_or_dataset, topk=100):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
            # 기존
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
            # 쿼리 전처리
                # doc_scores, doc_indices = self.get_relevant_doc_bulk([remove_double_space(remove_newlines(temp)) for temp in query_or_dataset['question']], k=topk)
            
            # print(doc_indices)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
            # 기존
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    # "context": self.contexts[doc_indices[idx][0]]  # retrieved doument            # 기본
                    'context' : ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)]) # 하나로 잇기
                    # 'context' : [self.contexts[doc_indices[idx][i]] for i in range(topk)]
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)
            # 변경
                # for i in range(topk) :
                #     tmp = {
                #         "question": example["question"],
                #         "id": example['id'],
                #         "context_id": doc_indices[idx][i],  # retrieved id
                #         # "context": self.contexts[doc_indices[idx][0]]  # retrieved doument            # 기본
                #         # 'context' : ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)]) # 하나로 잇기
                #         'context' : self.contexts[doc_indices[idx][i]]
                #     }
                #     if 'context' in example.keys() and 'answers' in example.keys():
                #         tmp["original_context"] = example['context']  # original document
                #         tmp["answers"] = example['answers']           # original answer
                #     total.append(tmp)
                    

            cqas = pd.DataFrame(total)
            # cqas.to_csv('./test.csv', sep = ',')
            # print(cqas)
            return cqas

    def get_relevant_doc(self, query, k=100):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=100):
        query_vec = self.tfidfv.transform(queries)
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(self, query_or_dataset, topk=5):
        assert self.indexer is not None, "You must build faiss by self.build_faiss() before you run self.retrieve_faiss()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset['question']
            # make retrieved result as dataframe
            total = []
            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]

                tmp = {
                    "question": example["question"],
                    "id": example['id'],  # original id
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"]: example['context']  # original document
                    tmp["answers"]: example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_faiss(self, query, k=5):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]


    def get_relevant_doc_bulk_faiss(self, queries, k=5):
        query_vecs = self.tfidfv.transform(queries)

        assert (
                np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()


if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("../input/data/data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents.json"
    retriever = SparseRetrieval(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="/opt/ml/input/data/data/",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    retriever.get_sparse_embedding()
    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query, 100)
    # with timer("single query by faiss"):
    #     scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds, 100)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search", df['correct'].sum() / len(df))
    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve_faiss(full_ds)
    #     df['correct'] = df['original_context'] == df['context']
    #     print("correct retrieval result by faiss", df['correct'].sum() / len(df))


