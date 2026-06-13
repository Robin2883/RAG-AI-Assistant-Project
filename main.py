from fastapi import FastAPI
import logging
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import pydantic
import os
import uvicorn
import datetime

from data_load import load_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom import RagChunkSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

load_dotenv()

inngest_client=inngest.Inngest(app_id='rag_app', logger=logging.getLogger("uvicorn"), is_production=False,serializer=inngest.PydanticSerializer())

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="app/ingest_pdf")
)

async def ingest_pdf(ctx: inngest.Context) -> str:
    def _load(ctx: inngest.Context) ->RagChunkSrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id", pdf_path)
        chunks=load_chunk_pdf(pdf_path)
        return RagChunkSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunksandsrc: RagChunkSrc) ->RAGUpsertResult:
        chunks=chunksandsrc.chunks
        source_id=chunksandsrc.source_id
        vecs=embed_texts(chunks)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads=[{"source":source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunksandsrc=await ctx.step.run("Load-chunk", lambda: _load(ctx), output_type=RagChunkSrc)
    ingested=await ctx.step.run ("embed-upsert", lambda: _upsert(chunksandsrc), output_type=RAGUpsertResult)
    return ingested.model_dump()

app=FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[ingest_pdf])



def main():
    print("Hello from ai-assistant!")


if __name__ == "__main__":
    main()
