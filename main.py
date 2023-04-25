from fastapi import FastAPI, File, HTTPException
from io import BytesIO
from pdfminer.high_level import extract_text
import openai
import pinecone
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import (
    create_engine,
    MetaData,
    Column,
    Integer,
    String,
    Float,
    Table,
)
from sqlalchemy.orm import sessionmaker
import uvicorn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

origins = [
    os.getenv("FRONTEND_URL"),
]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = create_engine(os.getenv("DATABASE_URL"))
Session = sessionmaker(bind=engine)
session = Session()


metadata = MetaData()

Job = Table(
    "Job",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("origin", String),
    Column("originId", String),
    Column("title", String),
    Column("company", String),
    Column("location", String),
    Column("jobDetails", String),
    Column("qualifications", String),
    Column("reviews", Integer),
    Column("stars", Float),
    Column("jobDescription", String),
    Column("benefits", String),
    Column("hiringInsights", String),
    Column("createdAt", String),
    Column("updatedAt", String),
)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
openai.api_key = os.getenv("OPENAI_API_KEY")

index = pinecone.Index("job")


@app.post("/")
def post_root(file: bytes = File()):
    try:
        with BytesIO(file) as pdf_file:
            text = extract_text(pdf_file)

        embedding = [x.item() for x in model.encode(text)]
        matches = [
            match._data_store
            for match in index.query(vector=embedding, top_k=10)["matches"]
        ]

        job_ids = [x["id"] for x in matches]
        jobs = session.query(Job).filter(Job.c.id.in_(job_ids)).all()

        jobs_dict = [job._asdict() for job in jobs]
        for job in jobs_dict:
            if job["origin"] == "ca.indeed.com":
                job["url"] = f"https://ca.indeed.com/viewjob?jk={job['originId']}"
        for match in matches:
            for job in jobs_dict:
                if match["id"] == job["id"]:
                    match.update(job)
        return sorted(matches, key=lambda x: x["score"], reverse=True)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/testing")
def testing():
    return os.getenv("FRONTEND_URL")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
