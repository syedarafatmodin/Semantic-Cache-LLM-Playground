import os
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class SemanticCacheQA:
    def __init__(self):
        # Initialize embedding model (Hugging Face)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq LLM with current model
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"  # Updated to current model
        )
        
        # Initialize Pinecone with new API (using free-tier supported region)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create or connect to Pinecone index
        self.index_name = "semantic-cache-qa"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Match embedding model dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free-tier supported region
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        
        # Similarity threshold (adjust as needed)
        self.similarity_threshold = 0.85
    
    def get_embedding(self, text):
        """Generate embedding for a given text"""
        return self.embedding_model.encode(text).tolist()
    
    def find_similar_question(self, query_embedding):
        """Search for similar questions in the cache"""
        results = self.index.query(
            vector=query_embedding,
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        
        if results.matches and results.matches[0].score > self.similarity_threshold:
            return results.matches[0]
        return None
    
    def generate_answer(self, question):
        """Generate answer using LLM"""
        try:
            response = self.llm.invoke(question)
            return response.content
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't generate an answer at this time."
    
    def add_to_cache(self, question, answer, embedding):
        """Store new Q&A pair in cache"""
        metadata = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
        # Corrected upsert syntax
        self.index.upsert(vectors=[{
            "id": str(hash(question)),
            "values": embedding,
            "metadata": metadata
        }])
    
    def ask_question(self, question):
        """Main method to handle questions"""
        print(f"\nQuestion: {question}")
        
        # Generate embedding for the question
        embedding = self.get_embedding(question)
        
        # Check cache for similar questions
        match = self.find_similar_question(embedding)
        
        if match:
            print(f"✅ Answer from cache (similarity: {match.score:.2f})")
            print(f"Matched question: {match.metadata['question']}")
            print(f"Answer: {match.metadata['answer']}")
            print(f"Originally cached at: {match.metadata['timestamp']}")
            return {
                "source": "cache",
                "answer": match.metadata["answer"],
                "similarity": match.score,
                "matched_question": match.metadata["question"],
                "timestamp": match.metadata["timestamp"]
            }
        else:
            # Generate new answer
            answer = self.generate_answer(question)
            
            # Add to cache
            self.add_to_cache(question, answer, embedding)
            
            print("🆕 Answer from LLM")
            print(f"Answer: {answer}")
            print(f"Cached at: {datetime.now().isoformat()}")
            return {
                "source": "llm",
                "answer": answer,
                "similarity": None,
                "matched_question": None,
                "timestamp": datetime.now().isoformat()
            }

# Initialize the QA system
qa_system = SemanticCacheQA()

# Test cases
test_sets = [
    {
        "name": "Oceans",
        "questions": [
            "How many oceans are there in the world?",
            "What is the count of oceans on Earth?"
        ]
    },
    {
        "name": "Capital",
        "questions": [
            "What is the capital of Japan?",
            "Which city is the capital of Japan?"
        ]
    },
    {
        "name": "India President",
        "questions": [
            "Who is the current President of the India?",
            "Who's leading the Indian government right now?"
        ]
    },
    {
        "name": "Cache Miss",
        "questions": [
            "What are the symptoms of flu?"
        ]
    }
]

# Run test cases
for test_set in test_sets:
    print(f"\n{'='*40}")
    print(f"Test Set: {test_set['name']}")
    print(f"{'='*40}")
    for question in test_set["questions"]:
        qa_system.ask_question(question)
        print("-" * 40)