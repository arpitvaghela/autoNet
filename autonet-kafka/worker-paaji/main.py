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

topicname = "training"


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

    while True:
        data = await consume(consumer, topicname)
        # response = ConsumerResponse(topic=topicname, **json.loads(data))
        response = str(data)
        print(f"worker:main[77]: {response}")
        if response == "train":
            print("GPU Available")
            print(torch.cuda.is_available())
            search("mnist-remote", "mnist", batch_size=16, epochs=1)


asyncio.run(func())
