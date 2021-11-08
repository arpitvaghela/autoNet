from routes.user import router as UserRouter
from routes.project import router as ProjectRouter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(debug=True)
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(UserRouter, tags=["User"], prefix="/user")

app.include_router(ProjectRouter, tags=["Project"], prefix="/project")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Autonet User & Project Service!"}
