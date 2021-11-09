import uuid
from datetime import datetime

from pydantic import BaseModel
from pydantic import validator
from pydantic.types import StrictStr


class TrainMessage(BaseModel):
    projectid: str
    dataid: str
    name: str
    dataset: str
    task: str = "train"
    timestamp: StrictStr = ""
    message_id: StrictStr = ""
    batch_size: int = 64
    w_lr: float = 0.025
    w_lr_min: float = 0.001
    w_momentum: float = 0.9
    w_weight_decay: float = 3e-4
    w_grad_clip: float = 5.0
    print_freq: int = 50
    gpus: str = "0"
    epochs: int = 50
    init_channels: int = 16
    layers: int = 8
    seed: int = 69
    workers: int = 4
    alpha_lr: float = 3e-4
    alpha_weight_decay: float = 1e-3

    @validator("message_id", pre=True, always=True)
    def set_id_from_name_uuid(cls, v, values):
        if "name" in values:
            return f"{values['name']}_{uuid.uuid4()}"
        else:
            raise ValueError("name not set")

    @validator("timestamp", pre=True, always=True)
    def set_datetime_utcnow(cls, v):
        return str(datetime.utcnow())


class TrainResponse(BaseModel):
    name: StrictStr
    message_id: StrictStr
    topic: StrictStr
    timestamp: StrictStr = ""

    @validator("timestamp", pre=True, always=True)
    def set_datetime_utcnow(cls, v):
        return str(datetime.utcnow())
