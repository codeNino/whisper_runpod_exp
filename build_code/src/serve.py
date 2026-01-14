from fastapi import FastAPI
import warnings
import subprocess
from voice_to_text import process_audio_request
from data_body import PayLoadData
from helpers import process_ml_agent
# import asyncio

# from contextlib import asynccontextmanager

# from config import BatchRequest, batch_queue
# from util import batch_worker

warnings.filterwarnings("ignore")

API_SUMMARY = """
This endpoint is designed to receive a batch of input instances, 
each containing a set of features. It processes these instances using a 
machine learning model and returns a corresponding set of predictions. 
The endpoint adheres to the SageMaker BYOC standard and requires a JSON request body 
"""

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # ---- startup ----
#     worker_task = asyncio.create_task(batch_worker())
#     yield
#     # ---- shutdown ----
#     worker_task.cancel()
#     try:
#         await worker_task
#     except asyncio.CancelledError:
#         pass

# app = FastAPI(title="MBL AI Models", summary=API_SUMMARY, lifespan=lifespan)

app = FastAPI(title="MBL AI Models", summary=API_SUMMARY)


@app.get("/ping")
def ping():
    return {"pong": 200}


@app.post("/invocations")
def invocations(payload: PayLoadData):
    if payload.is_ml_agent:
        return process_ml_agent(payload)
    else:
        return process_audio_request(
            payload.audio_url,
            payload.extra_data,
            payload.dispatcher_endpoint
        )


# @app.post("/invocations")
# async def invocations(payload: PayLoadData):
#     if payload.is_ml_agent:
#         return process_ml_agent(payload)
#     else:
#         loop = asyncio.get_event_loop()
#         future = loop.create_future()
#         await batch_queue.put(
#         BatchRequest(payload=payload, future=future)
#     )
#         return await future


def run_app():
    subprocess.run(["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"])


if __name__ == "__main__":
    run_app()
