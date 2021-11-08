
  
import motor.motor_asyncio
from bson.objectid import ObjectId

MONGO_DETAILS = "mongodb://root:rootpassword@mongodb_container:27017/"
# 

# MONGO_DETAILS = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.autonet

users_collection = database.get_collection("users_collection")
projects_collection = database.get_collection("projects_collection")

def user_helper(user) -> dict:
    return{
        "id": str(user["_id"]),
        "fullname": user["fullname"],
        "email": user["email"],
        "role":user["role"],
        "organisation":user["organisation"]
    }

def project_helper(project) -> dict:
    return{
        "id":str(project["_id"]),
        "projectname":project["projectname"],
        "userid":project["userid"],
        "dataid":project["dataid"],
        "datameta":project["datameta"]
    }

async def retrieve_users():
    users = []
    async for user in users_collection.find():
        users.append(user_helper(user))
    return users

async def retrieve_projects():
    projects = []
    async for project in projects_collection.find():
        projects.append(project_helper(project))
    return projects


async def add_user(user_data: dict) -> dict:
    user = await users_collection.insert_one(user_data)
    new_user = await users_collection.find_one({"_id": user.inserted_id})
    return user_helper(new_user)

async def add_project(project_data: dict) -> dict:
    project = await projects_collection.insert_one(project_data)
    new_project = await projects_collection.find_one({"_id": project.inserted_id})
    return project_helper(new_project)


async def retrieve_user(id: str) -> dict:
    user = await users_collection.find_one({"_id": ObjectId(id)})
    if user:
        return user_helper(user)

async def retrieve_project(id: str) -> dict:
    project = await projects_collection.find_one({"_id": ObjectId(id)})
    if project:
        return project_helper(project)

# Update a user with a matching ID
async def update_user(id: str, data: dict):
    # Return false if an empty request body is sent.
    print(id)
    if len(data) < 1:
        return False
    user = await users_collection.find_one({"_id": ObjectId(id)})
    if user:

        updated_user = await users_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_user:
            return True
        return False

async def update_project(id: str, data: dict):
    # Return false if an empty request body is sent.
    print(id)
    if len(data) < 1:
        return False
    project = await projects_collection.find_one({"_id": ObjectId(id)})
    if project:

        updated_project = await projects_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_project:
            return True
        return False

# Delete a user from the database
async def delete_user(id: str):
    user = await users_collection.find_one({"_id": ObjectId(id)})
    if user:
        await users_collection.delete_one({"_id": ObjectId(id)})
        return True

# Delete a user from the database
async def delete_project(id: str):
    project = await projects_collection.find_one({"_id": ObjectId(id)})
    if project:
        await projects_collection.delete_one({"_id": ObjectId(id)})
        return True
