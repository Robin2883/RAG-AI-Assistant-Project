from fastapi import FastAPI
import logging
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uvicorn
import datetime

from data_load import load_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

inngest_client=inngest.Inngest(app_id='rag_app', logger=logging.getLogger("uvicorn"), is_production=False,serializer=inngest.PydanticSerializer())

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="app/ingest_pdf")
)

async def ingest_pdf(ctx: inngest.Context) -> str:
    ctx.logger.info(ctx.event)
    return "done"

app=FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[ingest_pdf])



def main():
    print("Hello from ai-assistant!")


if __name__ == "__main__":
    main()
