import os
import requests
import zipfile
import io
import tempfile
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛑 Get these from cloud.qdrant.io
QDRANT_URL = os.getenv("QDRANT_URL", "https://bac8c5c6-52e9-43c8-bf77-dffe4cf7b2b8.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NzY3ZjQ1NjctZjhjMC00YmJlLWE2MWUtNDI4OTY4ODdmNTBjIn0.l19jv0VEQNl9_gAXsQRGyXXFznUW72eQQsBdz93KNEk")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Loading 80MB embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "icarus_github_context"

try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

class RepoPayload(BaseModel):
    repo_url: str

@app.post("/ingest")
async def ingest_repo(payload: RepoPayload):
    base_url = payload.repo_url.rstrip("/")
    zip_url = f"{base_url}/archive/refs/heads/main.zip"
    
    response = requests.get(zip_url)
    if response.status_code != 200:
        zip_url = f"{base_url}/archive/refs/heads/master.zip"
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download repo. Ensure it is public.")

    points = []
    point_id = 1
    allowed_exts = {".py", ".js", ".ts", ".html", ".css", ".md", ".json", ".txt"}
    
    # Hide the download from VS Code Live Server!
   # Hide the download from VS Code Live Server!
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 1. HERE IS THE IGNORE LIST
        ignore_dirs = {"node_modules", "venv", ".next", "dist", "build", "public", "assets", ".git"}
        architecture_map = []
        
        for root, dirs, files in os.walk(extract_dir):
            # Tell os.walk to skip our ignored directories entirely
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in allowed_exts:
                    file_path = os.path.join(root, file)
                    
                    # Save clean path to our architecture map
                    clean_path = file_path.replace(extract_dir, "")
                    architecture_map.append(clean_path) 
                    
                    # Skip files larger than 50KB to keep things fast
                    if os.path.getsize(file_path) > 50000:
                        continue
                        
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                        chunk_size = 800
                        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                        for chunk in chunks:
                            if not chunk.strip(): continue
                            vector = embedding_model.encode(chunk).tolist()
                            points.append(
                                PointStruct(
                                    id=point_id,
                                    vector=vector,
                                    payload={"path": clean_path, "text": chunk}
                                )
                            )
                            point_id += 1
                    except Exception:
                        pass 

        # 2. HERE IS THE MASTER MAP GENERATION
        if architecture_map:
            map_text = "PROJECT FILE DIRECTORY AND FOLDER STRUCTURE:\n" + "\n".join(architecture_map)
            map_vector = embedding_model.encode("folder structure directory file map architecture").tolist()
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=map_vector,
                    payload={"path": "MASTER_ARCHITECTURE_MAP.txt", "text": map_text}
                )
            )
            point_id += 1

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        
    return {"status": f"Successfully indexed {len(points)} chunks into Qdrant Cloud."}

@app.post("/query")
async def query_context(request: Request):
    data = await request.json()
    if not isinstance(data, dict):
        data = {}

    tool_call_id = "test_id"
    search_query = data.get("query", "")

    message = data.get("message")
    if isinstance(message, dict):
        tool_list = message.get("toolWithToolCallList")
        if isinstance(tool_list, list) and tool_list:
            first_tool = tool_list[0]
            if isinstance(first_tool, dict):
                tool_call = first_tool.get("toolCall")
                if isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id", tool_call_id)
                    function_data = tool_call.get("function")
                    if isinstance(function_data, dict):
                        raw_args = function_data.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                arguments = json.loads(raw_args)
                                if isinstance(arguments, dict):
                                    search_query = arguments.get("query", search_query)
                            except json.JSONDecodeError:
                                pass

    vector = embedding_model.encode(search_query).tolist()
    query_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=4
    )
    results = query_response.points
    
    if not results:
        context_str = "No relevant code found in the repository."
    else:
        context_parts = []
        for r in results:
            payload = r.payload if isinstance(r.payload, dict) else {}
            path = payload.get("path", "unknown")
            text = payload.get("text", "")
            context_parts.append(f"File: {path}\nCode: {text}")
        context_str = "\n\n".join(context_parts)
        
    return {
        "results": [
            {
                "toolCallId": tool_call_id,
                "result": f"Here is the code context from the user's repo:\n{context_str}"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
