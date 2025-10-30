from FlagEmbedding import BGEM3FlagModel
import umap
import hdbscan


class Model:
    def load_model(data):
        model = BGEM3FlagModel(data["MODEL"], local_files_only=True,
                               use_fp16=True)
        return model

    def load_umap(data):
        fit = umap.Umap(n_neighbors=data["UMAP"]["neighbors"],
                        min_dist=data["UMAP"]["dist"],
                        n_components=data["UMAP"]["components"],
                        metric=data["UMAP"]["metric"],
                        random_state=data["UMAP"]["random_state"])
        return fit

    def load_hdbscan(data):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=data["HDBSCAN"]["min_cluster_size"],
            min_samples=data["HDBSCAN"]["min_samples"],
            metric=data["HDBSCAN"]["metric"],
            cluster_selection_method=data["HDBSCAN"]
            ["cluster_selection_method"],
            prediction_data=data["HDBSCAN"][""]
        )
        return clusterer

    def __init__(self, data):
        self.data = data
        self.model = self.load_model()
        self.fit = self.load_umap()
        self.clusterer = self.load_hdbscan()

    def get_embeddings(self, corpus):
        embed = self.model.encode(corpus, batch_size=16,
                                  return_dense=True, return_sparse=False,
                                  return_colbert_vec=False)
        return embed["dense_vecs"]

    def change_size(self, embeddings):
        emb = self.umap.fit_transform(embeddings)
        return emb
