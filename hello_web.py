from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def home():
    return {"message":"欢迎来到我的第一个web应用"}


@app.get("/greet")
def great(name: str="陌生人"):
    return {"message":f"你好{name}"}

