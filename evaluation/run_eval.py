import json
from app.ingestion.loader import load_pdf, chunk_pages
from app.retrieval.retriever import Retriever

PDF = "data/sample.pdf"

r = Retriever(path="data/eval_chroma")
if r.col.count() == 0:
    r.add(chunk_pages(load_pdf(PDF)), source="sample.pdf")

tests = json.load(open("evaluation/test_set.json"))

configs = {
    "Vector only": dict(hybrid=False, rerank=False),
    "Hybrid (vector + BM25)": dict(hybrid=True, rerank=False),
    "Hybrid + reranker": dict(hybrid=True, rerank=True),
}

lines = ["| Pipeline | Hit rate @4 |", "|---|---|"]
for name, cfg in configs.items():
    hits = 0
    for t in tests:
        results = r.search(t["question"], k=4, **cfg)
        if any(x["page"] == t["page"] for x in results):
            hits += 1
    rate = 100 * hits / len(tests)
    lines.append(f"| {name} | {rate:.1f}% |")

table = "\n".join(lines)
print(table)
open("evaluation/results.md", "w").write(table)