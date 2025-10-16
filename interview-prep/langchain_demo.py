#!/usr/bin/env python3
"""
Simple LangChain Integration Demo
Demonstrates industry-standard RAG pipeline using LangChain

This shows how to:
1. Use LangChain for document processing
2. Create embeddings with LangChain
3. Build a simple retrieval chain
4. Integrate with OpenAI models
"""

import os
from typing import List
from dotenv import load_dotenv

# LangChain imports (simplified version)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock Document class for demonstration
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

load_dotenv()

class SimpleLangChainRAG:
    """
    Simple RAG implementation using LangChain
    Demonstrates industry-standard patterns
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.vector_store = None
        self.qa_chain = None
        
        if not LANGCHAIN_AVAILABLE:
            print("⚠️  LangChain not installed. This is a demonstration of the concepts.")
            print("   In production, you would: pip install langchain")
    
    def create_documents(self) -> List[Document]:
        """Create sample documents for the demo"""
        sample_texts = [
            """
            FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ 
            based on standard Python type hints. It provides automatic API documentation, data validation, 
            and serialization. FastAPI is built on top of Starlette and Pydantic, making it both fast and 
            developer-friendly.
            """,
            """
            Docker is a containerization platform that allows developers to package applications and their 
            dependencies into lightweight, portable containers. These containers can run consistently across 
            different environments, from development to production. Docker simplifies deployment, scaling, 
            and management of applications.
            """,
            """
            Kubernetes is an open-source container orchestration platform that automates the deployment, 
            scaling, and management of containerized applications. It provides features like service discovery, 
            load balancing, storage orchestration, and automated rollouts and rollbacks. Kubernetes is 
            essential for managing containers at scale.
            """,
            """
            Machine Learning is a subset of artificial intelligence that enables computers to learn and 
            make decisions from data without being explicitly programmed. It includes supervised learning, 
            unsupervised learning, and reinforcement learning. ML algorithms can identify patterns, make 
            predictions, and improve performance over time.
            """,
            """
            RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with 
            language generation. It retrieves relevant documents from a knowledge base and uses them to 
            generate more accurate and contextual responses. RAG helps reduce hallucinations in LLM outputs 
            and provides traceable information sources.
            """
        ]
        
        # Convert to LangChain Document objects
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text.strip(),
                metadata={"source": f"doc_{i}", "topic": ["FastAPI", "Docker", "Kubernetes", "ML", "RAG"][i]}
            )
            documents.append(doc)
        
        return documents
    
    def setup_rag_pipeline(self):
        """
        Set up the complete RAG pipeline using LangChain
        """
        if not LANGCHAIN_AVAILABLE:
            print("📚 LangChain Pipeline Setup (Conceptual)")
            print("1. Document Loading: ✅ 5 technical documents loaded")
            print("2. Text Splitting: ✅ RecursiveCharacterTextSplitter configured")
            print("3. Embeddings: ✅ OpenAI embeddings ready")
            print("4. Vector Store: ✅ FAISS vector database initialized")
            print("5. Retrieval Chain: ✅ RetrievalQA chain created")
            return
        
        print("🚀 Setting up LangChain RAG Pipeline...")
        
        # Step 1: Create documents
        documents = self.create_documents()
        print(f"📚 Loaded {len(documents)} documents")
        
        # Step 2: Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"✂️  Split into {len(splits)} chunks")
        
        # Step 3: Create embeddings
        if not self.api_key:
            print("⚠️  No OpenAI API key - using mock embeddings")
            return
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        print("🔢 Embeddings model initialized")
        
        # Step 4: Create vector store
        self.vector_store = FAISS.from_documents(splits, embeddings)
        print("💾 Vector store created with FAISS")
        
        # Step 5: Create retrieval chain
        llm = OpenAI(openai_api_key=self.api_key, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print("🔗 RetrievalQA chain created")
        print("✅ RAG Pipeline ready!")
    
    def query_rag_system(self, question: str) -> dict:
        """
        Query the RAG system with a question
        """
        if not LANGCHAIN_AVAILABLE or not self.qa_chain:
            # Mock response for demonstration
            return {
                "question": question,
                "answer": f"[Mock RAG Response] This would be an AI-generated answer based on retrieved documents about: {question}",
                "source_documents": [
                    {"content": "Mock document content...", "metadata": {"source": "doc_0"}},
                    {"content": "Another relevant document...", "metadata": {"source": "doc_1"}}
                ]
            }
        
        print(f"❓ Question: {question}")
        
        # Query the RAG system
        result = self.qa_chain({"query": question})
        
        # Format response
        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
        
        return response
    
    def demonstrate_rag_capabilities(self):
        """
        Demonstrate RAG capabilities with sample questions
        """
        print("\n" + "="*60)
        print("🧪 LangChain RAG Demonstration")
        print("="*60)
        
        # Set up the pipeline
        self.setup_rag_pipeline()
        
        # Sample questions
        questions = [
            "What is FastAPI and what are its main features?",
            "How does Docker help with application deployment?",
            "What is the difference between Docker and Kubernetes?",
            "Explain how RAG works and its benefits"
        ]
        
        print("\n📋 Testing RAG System with Sample Questions:")
        print("-" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            
            response = self.query_rag_system(question)
            
            print(f"💬 Answer: {response['answer'][:200]}...")
            print(f"📚 Sources: {len(response['source_documents'])} documents retrieved")
            
            # Show source information
            for j, source in enumerate(response['source_documents'][:2]):
                print(f"   Source {j+1}: {source['metadata'].get('topic', 'Unknown')} - {source['content'][:50]}...")
        
        print("\n" + "="*60)
        print("✅ LangChain RAG Demonstration Complete!")
        
        return {
            "pipeline_status": "ready" if LANGCHAIN_AVAILABLE and self.qa_chain else "mock",
            "questions_tested": len(questions),
            "langchain_available": LANGCHAIN_AVAILABLE
        }

def main():
    """
    Main demonstration function
    """
    print("🧪 LangChain Integration Demo")
    print("This demonstrates industry-standard RAG implementation")
    print()
    
    # Create and run the demo
    rag_demo = SimpleLangChainRAG()
    results = rag_demo.demonstrate_rag_capabilities()
    
    # Summary
    print("\n📊 Demo Summary:")
    print(f"   - Pipeline Status: {results['pipeline_status']}")
    print(f"   - Questions Tested: {results['questions_tested']}")
    print(f"   - LangChain Available: {results['langchain_available']}")
    
    if not results['langchain_available']:
        print("\n💡 To run with real LangChain:")
        print("   pip install langchain faiss-cpu")
        print("   export OPENAI_API_KEY=your_key_here")
    
    print("\n🎯 Key Takeaways for Interviews:")
    print("   ✅ Understand LangChain's modular architecture")
    print("   ✅ Know the RAG pipeline components")
    print("   ✅ Can explain document processing and retrieval")
    print("   ✅ Familiar with industry-standard tools")

if __name__ == "__main__":
    main()
