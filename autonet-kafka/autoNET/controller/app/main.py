import asyncio
import json

from aiokafka import AIOKafkaProducer
from app.core.config import KAFKA_INSTANCE
from app.core.config import PROJECT_NAME

from app.core.models.model import TrainMessage, TrainResponse

# from app.core.models.model import ProducerResponse
from fastapi import FastAPI, Request

from loguru import logger
import requests
from typing import List

app = FastAPI(title=PROJECT_NAME)

loop = asyncio.get_event_loop()
aioproducer = AIOKafkaProducer(
    loop=loop, client_id=PROJECT_NAME, bootstrap_servers=KAFKA_INSTANCE
)

# queue tasks logic
# Worker = namedtuple("Worker", ["id", "available"])


class Worker:
    def __init__(self, id, available):
        self.id = id
        self.available = available


topicname = "training"


class BackgroundQueueHandler:
    def __init__(self):
        self.queue = []
        self.workers: List[Worker] = []

    def find_available_worker(self):
        worker_path = "http://worker:5001/available"
        response = requests.request("GET", worker_path)
        msg = response.json()
        logger.info(msg)
        if msg["available"]:
            return 0
        return -1

    async def run(self):
        while True:
            # print("running queueing service")
            if len(self.queue) == 0:
                await asyncio.sleep(1)
                continue
            else:
                worker_idx = self.find_available_worker()
                print(worker_idx)
                if worker_idx == -1:
                    # no worker available wait till they are
                    logger.info("In No worker idx")
                    msgdict = self.queue.pop()
                    msgdict.update({"workerid": "not this worker"})
                    await aioproducer.send(
                        topicname, json.dumps(msgdict).encode("ascii")
                    )
                    self.queue.pop()
                    await asyncio.sleep(1)
                    continue

                logger.info("Here in working terrain")
                msgdict = self.queue.pop()
                msgdict.update({"workerid": self.workers[worker_idx].id})
                logger.info(f"sending msg to {self.workers[worker_idx].id}")
                r = await aioproducer.send(
                    topicname, json.dumps(msgdict).encode("ascii")
                )
                print(r)
                await asyncio.sleep(1)
                # self.workers[worker_idx].available = False


handler = BackgroundQueueHandler()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(handler.run())
    await aioproducer.start()


@app.on_event("shutdown")
async def shutdown_event():
    await aioproducer.stop()


@app.get("/worker/register/{id}")
async def register_worker(id: str, request: Request):
    # TODO: get worker path dynamically
    handler.workers.append(Worker(id, True))
    logger.info(handler.workers)
    return {"success": True, "message": "Added worker successfully!", "worker_id": id}


@app.post(f"/producer/{topicname}")
async def kafka_produce(msg: TrainMessage):
    """
    Produce a message into <topicname>
    This will produce a message into a Apache Kafka topic
    And this path operation will:
    * return ProducerResponse
    """
    # why are we not waiting for response here ??
    url = "http://userservice:8002/project/" + str(msg.projectid)
    response = requests.request("GET", url)
    responsejson = response.json()
    if responsejson["code"] != 200:
        return {"success": False, "code": 400, "message": "wrong project id"}

    if responsejson["data"][0]["dataid"] == "":
        return {
            "success": False,
            "code": 404,
            "message": "please provide data to train on",
        }

    msgdict = msg.dict()
    msgdict.update({"dataid": responsejson["data"][0]["dataid"]})
    logger.info(f"{msgdict}")
    # add request to handler queue
    handler.queue.append(msgdict)
    print(len(handler.queue))
    # await aioproducer.send(topicname, json.dumps(msgdict).encode("ascii"))
    # TODO: send back project and data ids
    response = TrainResponse(name=msg.name, message_id=msg.message_id, topic=topicname)
    logger.info(response)

    return response


@app.get("/ping")
def ping():
    return {"ping": "pong!"}
