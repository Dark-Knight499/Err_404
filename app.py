from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from rag import create_rag_chain
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
pdf_path = "sample.pdf" 
qa_chain = create_rag_chain(pdf_path)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    global qa_chain 
    qa_chain = create_rag_chain(file.filename)
    with open(file.filename, "wb") as f:
        f.write(contents)
    return {"message": "File uploaded successfully"}

@app.post("/query")
async def query(request: Request):
    text = (await request.json())['text']
    response = qa_chain.invoke({"query": text})
    return response['result']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
