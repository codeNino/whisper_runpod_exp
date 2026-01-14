import os
from dotenv import load_dotenv
from pathlib import Path

from dataclasses import dataclass
import asyncio
from typing import List, Any
from dataclasses import dataclass

MAX_BATCH_SIZE = 6          # tune based on GPU memory
BATCH_TIMEOUT = 0.07         # 70ms batching window

batch_queue: asyncio.Queue = asyncio.Queue()


dir_path = (Path(__file__) / ".." / ".." / "..").resolve()
env_path = os.path.join(dir_path, ".env")

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)


class Envs:
    HF_TOKEN = os.getenv("HF_TOKEN")
    COMPUTE_RATE_PER_SECOND = 0.0007


settings = Envs()


@dataclass
class BatchRequest:
    payload: Any
    future: asyncio.Future



