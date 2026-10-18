import requests
import streamlit as st

API = "http://localhost:8000"
st.title("📄 DocuMind Pro")

file = st.file_uploader("Upload a PDF", type="pdf")
if file and st.button("Process"):
    r = requests.post(f"{API}/upload", files={"file": (file.name, file.getvalue())})
    st.success(f"Indexed {r.json()['chunks']} chunks")

q = st.text_input("Ask a question about your documents")
if q:
    res = requests.post(f"{API}/ask", json={"question": q}).json()
    st.write(res["answer"])
    st.subheader("Sources")
    for s in res["sources"]:
        with st.expander(f"{s['source']}, page {s['page']}"):
            st.write(s["text"])