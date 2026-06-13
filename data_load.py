from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

splitter=SentenceSplitter(chunk_size=1000, chunk_overlap=50) #chunk_overlap means every chunk will keep the last 50 words of last chunk as its first items, so that relvent context is not lost

def load_chunk_pdf(path: str):
    docs=PDFReader().load_data(file=path)
    texts=[d.text for d in docs if getattr(d, "text", None)]
    chunks=[]
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts:list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(texts)
    return embedding


