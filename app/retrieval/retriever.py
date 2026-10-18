import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


class Retriever:
    def __init__(self, path="data/chroma"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection("docs")
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        data = self.col.get()
        self.ids = data["ids"]
        self.docs = {i: d for i, d in zip(data["ids"], data["documents"])}
        self.metas = {i: m for i, m in zip(data["ids"], data["metadatas"])}
        corpus = [self.docs[i].lower().split() for i in self.ids]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, chunks, source):
        start = self.col.count()
        self.col.add(
            ids=[f"{source}-{start + i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=self.embedder.encode([c["text"] for c in chunks]).tolist(),
            metadatas=[{"page": c["page"], "source": source} for c in chunks],
        )
        self._rebuild_bm25()

    def search(self, query, k=4, candidates=10, hybrid=True, rerank=True):
        if not self.ids:
            return []
        q_emb = self.embedder.encode([query]).tolist()
        vec_ids = self.col.query(
            query_embeddings=q_emb, n_results=min(candidates, len(self.ids))
        )["ids"][0]

        ranked = vec_ids
        if hybrid:
            scores = self.bm25.get_scores(query.lower().split())
            bm25_ids = [
                self.ids[i]
                for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:candidates]
            ]
            fused = {}  # Reciprocal Rank Fusion
            for lst in (vec_ids, bm25_ids):
                for rank, _id in enumerate(lst):
                    fused[_id] = fused.get(_id, 0) + 1 / (60 + rank)
            ranked = sorted(fused, key=fused.get, reverse=True)[:candidates]

        if rerank:
            scores = self.reranker.predict([(query, self.docs[i]) for i in ranked])
            ranked = [i for _, i in sorted(zip(scores, ranked), reverse=True)]

        return [{"text": self.docs[i], **self.metas[i]} for i in ranked[:k]]