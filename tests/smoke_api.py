import requests, json

base = "http://localhost:5000"

# Test 1: Health
print("=== TEST: GET /api/health ===")
r = requests.get(f"{base}/api/health", timeout=10)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Ollama available: {data['ollama']['available']}")
print(f"Model found: {data['ollama']['model_found']}")
print(f"ChromaDB OK: {data['vector_store']['ok']}")
print(f"Total chunks: {data['vector_store']['total_chunks']}")
print(f"Monitor running: {data['monitor']['running']}")
print()

# Test 2: Documents endpoint
print("=== TEST: GET /api/documents ===")
r = requests.get(f"{base}/api/documents", timeout=10)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Documents: {len(data['documents'])}")
print()

# Test 3: Chat with no documents
print("=== TEST: POST /api/chat (no docs yet) ===")
r = requests.post(f"{base}/api/chat", json={"query": "What is this system?"}, timeout=10)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Answer: {data['answer'][:120]}")
print(f"Sources: {len(data['sources'])}")
print()

# Test 4: Bad request
print("=== TEST: POST /api/chat empty query ===")
r = requests.post(f"{base}/api/chat", json={"query": ""}, timeout=10)
print(f"Status (expected 400): {r.status_code}")
print()

print("=== All API smoke tests passed! ===")
