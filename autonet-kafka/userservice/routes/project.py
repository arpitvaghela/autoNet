from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder

from database import (
    add_project,
    delete_project,
    retrieve_project,
    retrieve_projects,
    update_project,
)
from models.User import (
    ErrorResponseModel,
    ResponseModel,
)

from models.Project import (
    ProjectSchema,
    UpdateProjectSchema
)
router = APIRouter()


@router.post("/", response_description="project data added into the database")
async def add_project_data(project: ProjectSchema = Body(...)):
    print(project)
    project = jsonable_encoder(project)
    new_project = await add_project(project)
    return ResponseModel(new_project, "project added successfully.")


@router.get("/", response_description="projects retrieved")
async def get_projects():
    projects = await retrieve_projects()
    if projects:
        return ResponseModel(projects, "projects data retrieved successfully")
    return ResponseModel(projects, "Empty list returned")


@router.get("/{id}", response_description="project data retrieved")
async def get_project_data(id):
    project = await retrieve_project(id)
    if project:
        return ResponseModel(project, "project data retrieved successfully")
    return ErrorResponseModel("An error occurred.", 404, "project doesn't exist.")


@router.put("/{id}")
async def update_project_data(id: str, req: UpdateProjectSchema = Body(...)):

    req = {k: v for k, v in req.dict().items() if v is not None}
    print(req)
    updated_project = await update_project(id, req)
    if updated_project:
        return ResponseModel(
            "project with ID: {} name update is successful".format(id),
            "project name updated successfully",
        )
    return ErrorResponseModel(
        "An error occurred",
        404,
        "There was an error updating the project data.",
    )


@router.delete("/{id}", response_description="project data deleted from the database")
async def delete_project_data(id: str):
    deleted_project = await delete_project(id)
    if deleted_project:
        return ResponseModel(
            "project with ID: {} removed".format(
                id), "project deleted successfully"
        )
    return ErrorResponseModel(
        "An error occurred", 404, "project with id {0} doesn't exist".format(
            id)
    )
