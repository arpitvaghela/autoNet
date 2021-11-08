from typing import Optional

from pydantic import BaseModel,Field
from typing import Dict


class ProjectSchema(BaseModel):
    projectname: str = Field(...)
    userid:str=Field(...)
    dataid: Optional[str]
    datameta: Optional[Dict]


    # 0 entry not in pipeline 1 entry in pipeline

    class Config:
        schema_extra = {
            "example": {
                "name": "My project",
                "userid": "<your user id>",
            }
        }


class UpdateProjectSchema(BaseModel):
    projectname: Optional[str]
    userid: Optional[str]
    dataid: Optional[str]
    datameta: Optional[Dict]

    class Config:
        schema_extra = {
            "example": {
                "projectname": "My project",
                "userid": "<your user id>",
                "dataid":"<your data id>",
                "datameta":{
                    "type":0, # meta data type 0 = not required 1 = label info 2 = image info only
                    "description":"",
                    "label1":10,
                    "label2":10
                }
            }
        }
