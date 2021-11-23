import asyncio
import json
import logging
import typing

from aiokafka import AIOKafkaConsumer
from core.config import KAFKA_INSTANCE
from core.config import PROJECT_NAME
from loguru import logger
import torch
from darts.search import search
import uuid
import requests
from fastapi import FastAPI
import threading
import docker

topicname = "training"
workerid = None
available = True

def register_self():
    uid = uuid.uuid4()

    url = f"http://controller:8000/worker/register/{uid}"
    response = requests.request("GET", url)
    msg = response.json()
    logger.info(msg)
    global workerid
    workerid = msg["worker_id"]


async def consume(consumer, topicname):
    async for msg in consumer:
        return msg.value.decode()


async def func():
    loop = asyncio.get_event_loop()
    consumer = AIOKafkaConsumer(
        topicname,
        loop=loop,
        client_id=PROJECT_NAME,
        bootstrap_servers=KAFKA_INSTANCE,
        enable_auto_commit=False,
    )

    await consumer.start()

    register_self()

    while True:
        data = await consume(consumer, topicname)
        # response = ConsumerResponse(topic=topicname, **json.loads(data))
        print(data)

        response = json.loads(data)
        print(response["workerid"], workerid)
        if not response["workerid"] == workerid:
            logger.info(f"Skipping request directed to worker: {response['workerid']}")
            continue
        try:
            del response["timestamp"]
            del response["message_id"]
            del response["workerid"]
        except:
            pass
        if "task" in response and response["task"] == "train":
            del response["task"]
            print("GPU Available")
            print(torch.cuda.is_available())
            await train(response)
            #global available
            # available = False
            # search(**response)
            # available = True

async def train(args):
    t = threading.Thread(target=search,kwargs=args)
    t.start()
    global available
    available = False
    while t.is_alive():
        await asyncio.sleep(5)
    available = True    
        
# asyncio.run(func())

app = FastAPI(title="worker")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(func())


@app.get("/available")
async def ping():
    global available
    return {"available": available}


@app.get("/")
async def root():
    return "Lord worker said Hello!"
