import requests, time, json

base = "http://localhost:5000"

print("=== END-TO-END RAG TEST ===")
print()

# Step 1: Trigger rescan so the sample_overview.txt gets indexed
print("[1] Triggering rescan to index sample_overview.txt...")
r = requests.post(f"{base}/api/documents/rescan", timeout=10)
print(f"    Rescan status: {r.status_code} — {r.json().get('message')}")

# Step 2: Wait for indexing to complete
print("[2] Waiting 15 seconds for indexing...")
time.sleep(15)

# Step 3: Check documents are indexed
print("[3] Checking indexed documents...")
r = requests.get(f"{base}/api/documents", timeout=10)
data = r.json()
print(f"    Documents: {len(data['documents'])}")
print(f"    Total chunks: {data['total_chunks']}")
for doc in data["documents"]:
    print(f"    - {doc['filename']} | status={doc['status']} | chunks={doc['chunk_count']}")
print()

if data["total_chunks"] == 0:
    print("ERROR: No chunks indexed. Stopping test.")
    exit(1)

# Step 4: Ask a real question
question = "What file types are supported by the system?"
print(f"[4] Asking question: '{question}'")
r = requests.post(f"{base}/api/chat", json={"query": question}, timeout=120)
resp = r.json()

print()
print("=== ANSWER ===")
print(resp["answer"])
print()
print("=== SOURCES ===")
for s in resp["sources"]:
    page_str = f" (page {s['page']})" if s.get("page") else ""
    print(f"  [{s['filename']}{page_str}] {s['excerpt'][:100]}...")
print()
print(f"Chunks retrieved: {resp['chunks_found']}")
print(f"Model: {resp['model']}")
print()

# Step 5: Second question
question2 = "How does the hybrid retrieval work?"
print(f"[5] Asking question: '{question2}'")
r2 = requests.post(f"{base}/api/chat", json={"query": question2}, timeout=120)
resp2 = r2.json()
print("=== ANSWER ===")
print(resp2["answer"])
print()
print("=== END-TO-END TEST COMPLETE ===")
