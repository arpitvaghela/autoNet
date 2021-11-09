import asyncio
import json

from aiokafka import AIOKafkaProducer
from app.core.config import KAFKA_INSTANCE
from app.core.config import PROJECT_NAME

from app.core.models.model import TrainMessage, TrainResponse

# from app.core.models.model import ProducerResponse
from fastapi import FastAPI

from loguru import logger
import requests

app = FastAPI(title=PROJECT_NAME)

loop = asyncio.get_event_loop()
aioproducer = AIOKafkaProducer(
    loop=loop, client_id=PROJECT_NAME, bootstrap_servers=KAFKA_INSTANCE
)


@app.on_event("startup")
async def startup_event():
    await aioproducer.start()


@app.on_event("shutdown")
async def shutdown_event():
    await aioproducer.stop()


@app.post("/producer/{topicname}")
async def kafka_produce(msg: TrainMessage, topicname: str):
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

    msgjson = json.loads(msg.json()).update({"dataid": responsejson["data"][0]["dataid"]})
    # request worker to search for projectid
    await aioproducer.send(topicname, json.dumps(msgjson).encode("ascii"))
    # TODO: send back project and data ids
    response = TrainResponse(name=msg.name, message_id=msg.message_id, topic=topicname)
    logger.info(response)

    return response


@app.get("/ping")
def ping():
    return {"ping": "pong!"}
