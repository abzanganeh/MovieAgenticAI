from services.rag import get_retriever

print("Testing RAG setup...")
retriever = get_retriever()

# Test query
results = retriever.invoke("action movie from the 90s")

print("\nTest Results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.metadata['title']}")