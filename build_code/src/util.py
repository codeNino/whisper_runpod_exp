from config import MAX_BATCH_SIZE, BATCH_TIMEOUT, batch_queue, BatchRequest
from voice_to_text import process_audio_request

import asyncio
from typing import List




def process_audio_batch(batch: List[BatchRequest]):
    results = []
    for req in batch:

        result = process_audio_request(
            req.payload.audio_url,
            req.payload.extra_data,
            req.payload.dispatcher_endpoint
        )
        results.append(result)
    return results



async def batch_worker():
    while True:
        batch: List[BatchRequest] = []

        # Always wait for at least one request
        first = await batch_queue.get()
        batch.append(first)

        start_time = asyncio.get_event_loop().time()

        # Collect more requests within timeout
        while len(batch) < MAX_BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (
                asyncio.get_event_loop().time() - start_time
            )
            if remaining <= 0:
                break

            try:
                item = await asyncio.wait_for(
                    batch_queue.get(), timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # ---- RUN BATCH INFERENCE ----
        try:
            results = process_audio_batch(batch)
            for req, result in zip(batch, results):
                req.future.set_result(result)
        except Exception as e:
            for req in batch:
                req.future.set_exception(e)
