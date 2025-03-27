import os
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone  # Correct import
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class SemanticCacheQA:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )
        
        # Initialize Pinecone with correct package
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        self.index_name = "semantic-cache-qa"
        self._initialize_index()
        self.similarity_threshold = 0.85

    def _initialize_index(self):
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        self.index = self.pc.Index(self.index_name)

    def get_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    def find_similar_question(self, query_embedding):
        results = self.index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        if results.matches and results.matches[0].score > self.similarity_threshold:
            return results.matches[0]
        return None

    def generate_answer(self, question):
        try:
            response = self.llm.invoke(question)
            return response.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def add_to_cache(self, question, answer, embedding):
        metadata = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        self.index.upsert(vectors=[{
            "id": str(hash(question)),
            "values": embedding,
            "metadata": metadata
        }])

    async def ask_question(self, question):
        embedding = self.get_embedding(question)
        match = self.find_similar_question(embedding)
        
        if match:
            return {
                "source": "cache",
                "answer": match.metadata["answer"],
                "similarity": float(match.score),
                "matched_question": match.metadata["question"],
                "timestamp": match.metadata["timestamp"]
            }
        else:
            answer = self.generate_answer(question)
            self.add_to_cache(question, answer, embedding)
            return {
                "source": "llm",
                "answer": answer,
                "similarity": None,
                "matched_question": None,
                "timestamp": datetime.now().isoformat()
            }

# Initialize the QA system
qa_system = SemanticCacheQA()

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        return await qa_system.ask_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)