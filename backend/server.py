import os
import math
import re
import hashlib
from collections import Counter
from dotenv import load_dotenv
import requests
import zipfile
import io
import tempfile
import json
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Keep local dev origins explicit for browser preflight reliability.
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛑 Get these from cloud.qdrant.io
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VAPI_PUBLIC_KEY = os.getenv("VAPI_PUBLIC_KEY")
LEGACY_VAPI_KEY = os.getenv("VAPI_KEY")
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")
missing = []
if not QDRANT_URL:
    missing.append("QDRANT_URL")
if not QDRANT_API_KEY:
    missing.append("QDRANT_API_KEY")
if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}. Add them to .env or set them in the environment.")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "icarus_github_context"
ACTIVE_REPO_KEY = None
ACTIVE_ERROR_CONTEXT = ""

try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def _ensure_repo_key_index() -> None:
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="repo_key",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except Exception as e:
        # Keep service up; ingest path will retry and return a clear error if still failing.
        print(f"Warning: could not ensure payload index for repo_key: {e}")


_ensure_repo_key_index()

class RepoPayload(BaseModel):
    repo_url: str


class ErrorContextPayload(BaseModel):
    text: str = ""


def _embed_text(text: str, dim: int = 384) -> list[float]:
    # Offline hash embedding to avoid external model downloads.
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    if not tokens:
        tokens = [lowered[:64] or "_empty_"]

    vector = [0.0] * dim
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        weight = 1.0 + math.log1p(count)

        idx1 = int.from_bytes(digest[:4], "big") % dim
        sign1 = 1.0 if (digest[4] & 1) == 0 else -1.0
        vector[idx1] += sign1 * weight

        idx2 = int.from_bytes(digest[5:9], "big") % dim
        sign2 = 1.0 if (digest[9] & 1) == 0 else -1.0
        vector[idx2] += sign2 * (0.5 * weight)

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


def _repo_key_from_url(repo_url: str) -> str:
    parsed = urlparse(repo_url.strip())
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{host}/{path}".strip("/")


def _resolve_github_archive_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]

    # Fast path for standard GitHub repo URLs.
    if parsed.netloc.lower() in {"github.com", "www.github.com"} and len(path_parts) >= 2:
        owner, repo = path_parts[0], path_parts[1].replace(".git", "")
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        try:
            api_response = requests.get(
                api_url,
                headers={"Accept": "application/vnd.github+json"},
                timeout=15,
            )
            if api_response.status_code == 200:
                default_branch = api_response.json().get("default_branch")
                if default_branch:
                    return f"https://github.com/{owner}/{repo}/archive/refs/heads/{default_branch}.zip"
        except Exception:
            pass

    # Fallback if API metadata is unavailable or URL is non-standard.
    for branch in ("main", "master", "dev", "develop", "trunk"):
        candidate = f"{base_url}/archive/refs/heads/{branch}.zip"
        try:
            response = requests.get(candidate, timeout=20)
            if response.status_code == 200:
                return candidate
        except Exception:
            continue

    return ""

@app.get("/config")
async def get_config():
    if not VAPI_ASSISTANT_ID:
        raise HTTPException(
            status_code=500,
            detail="Missing Vapi configuration. Set VAPI_ASSISTANT_ID in .env.",
        )

    if not VAPI_PUBLIC_KEY and LEGACY_VAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "VAPI_PUBLIC_KEY is required for web calls. "
                "You currently set VAPI_KEY. Set VAPI_PUBLIC_KEY to your Vapi "
                "public key and restart the backend."
            ),
        )

    if not VAPI_PUBLIC_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing Vapi configuration. Set VAPI_PUBLIC_KEY and VAPI_ASSISTANT_ID in .env.",
        )

    return {
        "publicKey": VAPI_PUBLIC_KEY,
        "assistantId": VAPI_ASSISTANT_ID,
    }


@app.post("/context/error")
async def set_error_context(payload: ErrorContextPayload):
    global ACTIVE_ERROR_CONTEXT
    ACTIVE_ERROR_CONTEXT = (payload.text or "").strip()[:12000]
    return {"status": "ok", "chars": len(ACTIVE_ERROR_CONTEXT)}

@app.post("/ingest")
async def ingest_repo(payload: RepoPayload):
    global ACTIVE_REPO_KEY
    base_url = payload.repo_url.rstrip("/")
    if base_url.endswith(".git"):
        base_url = base_url[:-4]
    repo_key = _repo_key_from_url(base_url)
    
    points = []
    import uuid

    stats = {
        "repo": repo_key,
        "scanned_files": 0,
        "eligible_files": 0,
        "indexed_files": 0,
        "skipped_large_files": 0,
        "skipped_filtered_files": 0,
        "read_errors": 0,
        "chunks_indexed": 0,
    }

    # Re-indexing should replace prior context for this repository.
    delete_filter = Filter(
        must=[
            FieldCondition(
                key="repo_key",
                match=MatchValue(value=repo_key),
            )
        ]
    )
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=delete_filter,
            wait=True,
        )
    except Exception:
        # Recover automatically if collection exists but payload index was missing.
        _ensure_repo_key_index()
        try:
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=delete_filter,
                wait=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear existing points for repo '{repo_key}': {e}",
            )
    
    if "huggingface.co" in base_url:
        raise HTTPException(
            status_code=400,
            detail="Hugging Face URLs are not supported in this offline mode. Please use a public GitHub repository URL.",
        )
    
    # Otherwise treat as GitHub Repo
    zip_url = _resolve_github_archive_url(base_url)
    if not zip_url:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve repository archive URL. Ensure it is a public GitHub repository.",
        )

    try:
        response = requests.get(zip_url, timeout=45)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download repository archive: {e}")
    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Could not download repository archive (status {response.status_code}). Ensure it is public.",
        )

    allowed_exts = {
        ".py", ".js", ".jsx", ".ts", ".tsx",
        ".html", ".css", ".scss",
        ".md", ".mdx", ".txt", ".rst",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".env",
        ".java", ".kt", ".kts", ".scala",
        ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
        ".rb", ".php", ".swift", ".sql", ".sh", ".bash", ".zsh",
        ".vue", ".svelte",
    }
    ignored_file_names = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "Cargo.lock",
    }
    max_file_size_bytes = 300_000
    chunk_size = 1200
    chunk_overlap = 200
    step = max(1, chunk_size - chunk_overlap)
    
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_dir)
        
        ignore_dirs = {
            "node_modules",
            "venv",
            ".venv",
            ".next",
            "dist",
            "build",
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".turbo",
            ".cache",
        }
        architecture_map = []
        
        for root, dirs, files in os.walk(extract_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                stats["scanned_files"] += 1
                ext = os.path.splitext(file)[1]
                if ext not in allowed_exts or file in ignored_file_names or file.endswith(".min.js"):
                    stats["skipped_filtered_files"] += 1
                    continue

                stats["eligible_files"] += 1
                file_path = os.path.join(root, file)
                clean_path = file_path.replace(extract_dir, "")
                architecture_map.append(clean_path)

                if os.path.getsize(file_path) > max_file_size_bytes:
                    stats["skipped_large_files"] += 1
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    stats["read_errors"] += 1
                    continue

                if not content.strip():
                    continue

                file_had_chunks = False
                for i in range(0, len(content), step):
                    chunk = content[i:i + chunk_size]
                    if not chunk.strip():
                        continue
                    vector = _embed_text(chunk)
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={"repo_key": repo_key, "path": clean_path, "text": chunk}
                        )
                    )
                    stats["chunks_indexed"] += 1
                    file_had_chunks = True

                if file_had_chunks:
                    stats["indexed_files"] += 1

        if architecture_map:
            max_map_entries = 5000
            map_lines = architecture_map[:max_map_entries]
            truncation_note = ""
            if len(architecture_map) > max_map_entries:
                truncation_note = f"\n... and {len(architecture_map) - max_map_entries} more files."
            map_text = (
                "PROJECT FILE DIRECTORY AND FOLDER STRUCTURE:\n"
                + "\n".join(map_lines)
                + truncation_note
            )
            map_vector = _embed_text("folder structure directory file map architecture")
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=map_vector,
                    payload={
                        "repo_key": repo_key,
                        "path": "MASTER_ARCHITECTURE_MAP.txt",
                        "text": map_text,
                    }
                )
            )
            stats["chunks_indexed"] += 1

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        ACTIVE_REPO_KEY = repo_key

    return {
        "status": f"Successfully indexed {len(points)} chunks into Qdrant Cloud.",
        "stats": stats,
        "activeRepo": ACTIVE_REPO_KEY,
    }

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

    search_query = (search_query or "").strip()
    if not search_query:
        search_query = "high level summary of this repository"

    vector = _embed_text(search_query)
    repo_filter = None
    if ACTIVE_REPO_KEY:
        repo_filter = Filter(
            must=[
                FieldCondition(
                    key="repo_key",
                    match=MatchValue(value=ACTIVE_REPO_KEY),
                )
            ]
        )
    query_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=repo_filter,
        limit=8
    )
    results = query_response.points or []

    # Always return best available context. A strict threshold can hide useful snippets
    # for short or oddly-phrased queries.
    if not results:
        context_str = (
            f"User query: {search_query}\n"
            f"Active repository: {ACTIVE_REPO_KEY or 'none'}\n"
            "Repository context: no snippets were retrieved.\n"
            "Assistant instructions:\n"
            "- Do NOT reply with generic refusals like 'I need more repository context'.\n"
            "- State briefly that indexing may be incomplete, then provide a best-effort answer.\n"
            "- End the response after the best-effort answer.\n"
            "- Do NOT ask for more context unless the user explicitly asks for deeper verification."
        )
    else:
        context_parts = []
        for r in results:
            payload = r.payload if isinstance(r.payload, dict) else {}
            path = payload.get("path", "unknown")
            text = payload.get("text", "")
            score = getattr(r, "score", None)
            score_line = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            context_parts.append(f"File: {path}\nRelevance: {score_line}\nCode: {text}")
        context_str = "\n\n".join(context_parts)

    if ACTIVE_ERROR_CONTEXT:
        context_str = (
            "Runtime stack trace and terminal logs from user session "
            "(use this as primary debugging context):\n"
            f"{ACTIVE_ERROR_CONTEXT[:4000]}\n\n"
            f"{context_str}"
        )

    return {
        "results": [
            {
                "toolCallId": tool_call_id,
                "result": (
                    "Voice response contract (must follow):\n"
                    "- Sound like a teammate speaking naturally, not reading a document.\n"
                    "- Use plain conversational sentences only. No headings, no bullet lists, no section labels.\n"
                    "- Keep answers tight: usually 2-4 sentences.\n"
                    "- Start with the answer immediately; no preamble.\n"
                    "- Do not ask for more context in normal flow.\n"
                    "- Do not read raw file paths, slash-separated names, or relevance scores aloud unless the user explicitly asks.\n"
                    "- When asked about architecture, summarize responsibilities and flow, not a directory recital.\n\n"
                    "Grounding context for reasoning (do not read verbatim):\n"
                    f"{context_str}"
                )
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
