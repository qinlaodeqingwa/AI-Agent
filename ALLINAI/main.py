from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
app = FastAPI()
users= []
class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users",response_model=List[User])
def get_users():
    return users
@app.post("/users",response_model=User)
def create_user(user: User):
    users.append(user)
    return user
@app.put("/users/{user_id}",response_model=User)
def update_user(user_id: int,user: User):
    for index,existing_user in enumerate(users):
        if existing_user.id == user_id:
            users[index] = user
            return user
    raise HTTPException(status_code=404, detail="User not found")
@app.delete("/users/{user_id}",response_model=User)
def delete_user(user_id: int):
    for index,user in enumerate(users):
        if user.id == user_id:
            users.pop(index)
            return {"message":"User deleted"}
    raise HTTPException(status_code=404, detail="User not found")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8596)