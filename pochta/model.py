from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from umap.umap_ import UMAP
import re
import hdbscan
import pandas as pd
from tqdm import tqdm
import numpy as np
import spacy

import json

class Model:
    def load_model(self,data):
        model = BGEM3FlagModel(data["MODEL"], local_files_only=True,
                               use_fp16=True)
        return model

    def load_umap(self,data):
        fit = UMAP(n_neighbors=data["UMAP"]["neighbors"],
                        min_dist=data["UMAP"]["dist"],
                        n_components=data["UMAP"]["components"],
                        metric=data["UMAP"]["metric"],
                        random_state=data["UMAP"]["random_state"])
        return fit
    
    def load_spacy(self):
        nlp = spacy.load("ru_core_news_md")
        return nlp

    def load_hdbscan(self,data):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=data["HDBSCAN"]["min_cluster_size"],
            min_samples=data["HDBSCAN"]["min_samples"],
            metric=data["HDBSCAN"]["metric"],
            cluster_selection_method=data["HDBSCAN"]
            ["cluster_selection_method"],
            prediction_data=data["HDBSCAN"]["prediction_data"]
        )
        return clusterer
    
    def load_vectorizer(self,data):
        vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(2, 4), 
        min_df=2,  
        max_df=0.8,  
        token_pattern=r"(?u)\b[а-яё]{2,}\b"
        )
        return vectorizer
    
    def load_dataset(self,data):
        df = pd.read_excel(data["DATASET"], sheet_name=0)
        return df

    def __init__(self, data):
        self.data = data
        self.model = self.load_model(data)
        self.umap = self.load_umap(data)
        self.clusterer = self.load_hdbscan(data)
        self.dataset=self.load_dataset(data)
        self.vectorizer = self.load_vectorizer(data)
        self.nlp = self.load_spacy()

    def normalize_text(self,text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub("ѣ", "е", text)
        text = re.sub("і", "и", text)
        text = re.sub("ѳ", "ф", text)
        text = re.sub("ъ(?=\s|$)", "", text)
        text = re.sub("[^а-яё\s]", " ", text)
        text = re.sub("\s+", " ", text).strip()
        return text

    def get_embeddings(self, corpus):
        embed = self.model.encode(corpus, batch_size=16,
                                  return_dense=True, return_sparse=False)
        return embed["dense_vecs"]

    def change_size(self, embeddings):
        emb = self.umap.fit_transform(embeddings)
        return emb
    
    def get_pos_pattern(self,phrase):
        doc = self.nlp(phrase)
        return " ".join([t.pos_ for t in doc if not t.is_punct])
    
    def get_cluster_cores(self, embeddings, df_phr, labels, probs):
        core_rows = []
        for cid in sorted(set(labels) - {-1}):
            idx = np.where(labels == cid)[0]
            emb_subset = embeddings[idx]
            center = emb_subset.mean(axis=0, keepdims=True)
            sims = cosine_similarity(emb_subset, center).ravel()
            core_idx = idx[sims.argmax()]
            core_rows.append({
                "cluster": cid,
                "core_phrase": df_phr.iloc[core_idx]["phrase"],
                "pattern": df_phr.iloc[core_idx]["pattern"],
                "count": len(idx),
                "mean_prob": float(probs[idx].mean())
            })
        core_df = pd.DataFrame(core_rows).sort_values(["count", "mean_prob"], ascending=False)
        return core_df
    
    def sep_by_language(self, language):
        df_lang = (
        self.dataset[self.dataset["Язык текста открытки"] == language]
        .dropna(subset=["Текст открытки"])
        .reset_index(drop=True)
        )

        if df_lang.empty:
            return
        df_lang["text_clean"] = df_lang["Текст открытки"].apply(self.normalize_text)

        X = self.vectorizer.fit_transform(df_lang["text_clean"])
        phrases = self.vectorizer.get_feature_names_out()
        frequencies = X.sum(axis=0).A1

        pos_patterns = []
        for phrase in tqdm(phrases):
            pos_patterns.append(self.get_pos_pattern(phrase))

        df_phr = pd.DataFrame({
            "phrase": phrases, 
            "freq": frequencies, 
            "pattern": pos_patterns
        })

        df_phr_filtered = df_phr[df_phr["pattern"].isin(self.data["TQDM"]["keep_patterns"])].copy()

        meaningful_phrases = df_phr_filtered['phrase'].tolist()

        embeddings = self.get_embeddings(meaningful_phrases)
        emb = self.change_size(embeddings)

        labels = self.clusterer.fit_predict(emb)
        probs = self.clusterer.probabilities_

        df_phr_filtered["cluster"] = labels
        df_phr_filtered["prob"] = probs

        core_df = self.get_cluster_cores(embeddings, df_phr_filtered, labels, probs)

        df_phr_filtered.to_csv(f"C:/Users/burov/Documents/pochta/pochta/pochta/results/{language}_phrases_with_clusters.csv", index=False)
        core_df.to_csv(f"C:/Users/burov/Documents/pochta/pochta/pochta/results/{language}_cluster_cores.csv", index=False)

    

    def build_json(self):
        """
        Собирает общий JSON-объект:
        язык → кластер → фразы
        """
        tree = {}

    
        for language in self.dataset["Язык текста открытки"].dropna().unique():
            try:
                df_phr = pd.read_csv(f"C:/Users/burov/Documents/pochta/pochta/pochta/results/{language}_phrases_with_clusters.csv")
                df_core = pd.read_csv(f"C:/Users/burov/Documents/pochta/pochta/pochta/results/{language}_cluster_cores.csv")
            except FileNotFoundError:
                continue

            clusters = []


            for _, row in df_core.iterrows():
                cluster_id = int(row["cluster"])


                cluster_phrases = df_phr[df_phr["cluster"] == cluster_id][
                    ["phrase", "freq", "prob"]
                ].to_dict(orient="records")

                cluster_obj = {
                    "cluster_id": cluster_id,
                    "core_phrase": row["core_phrase"],
                    "pattern": row["pattern"],
                    "count": int(row["count"]),
                    "mean_prob": float(row["mean_prob"]),
                    "phrases": cluster_phrases
                }
                clusters.append(cluster_obj)

            tree[language] = clusters

        output_path = "C:/Users/burov/Documents/pochta/pochta/pochta/results/tree.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)

        return tree



    def make_tree(self):
        n = 1
        for language in self.dataset["Язык текста открытки"].unique():
            self.sep_by_language(language)
            self.build_json()
            if n == 1:
                break
        
        

