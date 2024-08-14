from fastapi import FastAPI, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from app.helper import prepare_qa_chain
import uvicorn

app = FastAPI()


class QueryModel(BaseModel):
    query: str


# Initialize QA chain once and cache the result
@lru_cache(maxsize=1)
def get_qa_chain():
    return prepare_qa_chain()


qa_chain = get_qa_chain()


@app.post("/query/")
async def query_qa_system(query: QueryModel):
    user_input = query.query
    try:
        result = qa_chain.invoke({"query": user_input})
        answer_text = result['result'].split('Answer:')[-1].strip()

        if not answer_text:
            raise HTTPException(status_code=404, detail="Answer not found")

        return {"response": answer_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/")
async def welcome():
    return {"message": "Welcome to the PsychChatMistral API"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, reload=True)


