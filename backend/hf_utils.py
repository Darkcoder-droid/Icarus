import requests
import re

def parse_huggingface_repo(repo_url: str):
    """
    Fetches details from a Hugging Face model repository.
    Explains what kind of model it is and what quantizations are available.
    """
    repo_url = repo_url.rstrip("/")
    # Extract the repo id from the url (e.g., "TheBloke/Llama-2-7b-Chat-GGUF")
    # A standard URL looks like https://huggingface.co/username/repo
    # Handle if it's a dataset or model
    is_dataset = "/datasets/" in repo_url.lower()
    
    # Split by domain and get everything after
    repo_path = repo_url.split("huggingface.co/")[-1]
    # Remove query parameters and hashes
    repo_path = repo_path.split("?")[0].split("#")[0]
    # Split into components and filter out empty strings (like trailing slashes or /tree/main)
    parts = [p for p in repo_path.split("/") if p and p not in ("tree", "blob", "datasets")]
    
    if len(parts) >= 2:
        repo_id = f"{parts[0]}/{parts[1]}"
    elif len(parts) == 1:
        repo_id = parts[0]
    else:
        raise ValueError("Could not determine repository ID from URL.")
    
    # Select correct API endpoint
    api_type = "datasets" if is_dataset else "models"
    api_url = f"https://huggingface.co/api/{api_type}/{repo_id}"
    
    print(f"Querying HF API: {api_url}")
    resp = requests.get(api_url)
    
    if resp.status_code != 200:
        raise ValueError(f"Could not fetch data for Hugging Face repo: {repo_id}. Status code: {resp.status_code}")
        
    data = resp.json()
    
    pipeline_tag = str(data.get("pipeline_tag") or "Unknown")
    tags = data.get("tags") or []
    author = data.get("author", "Unknown")
    downloads = data.get("downloads", 0)
    likes = data.get("likes", 0)
    
    siblings = data.get("siblings", [])
    files = [sib.get("rfilename", "") for sib in siblings]
    
    quantizations = set()
    model_formats = set()
    model_files = []
    
    for fname in files:
        fname_lower = fname.lower()
        if fname_lower.endswith(".gguf"):
            model_formats.add("GGUF")
            model_files.append(fname)
            # Try to extract the quantization level from the filename, e.g. Q4_K_M
            # Usually it's something like model.Q4_K_M.gguf or Q4_K.gguf
            match = re.search(r'(q[0-8]_[a-z0-9_]+)\.gguf$', fname_lower, re.IGNORECASE)
            if match:
                quantizations.add(match.group(1).upper())
            else:
                quantizations.add("GGUF-Quantized")
        elif fname_lower.endswith(".awq") or "awq" in fname_lower:
            model_formats.add("AWQ")
            quantizations.add("AWQ")
        elif "gptq" in fname_lower:
            model_formats.add("GPTQ")
            quantizations.add("GPTQ")
        elif fname_lower.endswith(".safetensors"):
            model_formats.add("Safetensors")
            model_files.append(fname)
        elif fname_lower.endswith(".bin"):
            model_formats.add("PyTorch (.bin)")
            model_files.append(fname)
        elif fname_lower.endswith(".onnx"):
            model_formats.add("ONNX")
            model_files.append(fname)

    # Building a comprehensive text breakdown of the model
    text_content = f"Repository: {repo_id} (Author: {author})\n"
    text_content += f"Type of AI Model: {pipeline_tag.capitalize().replace('-', ' ')}\n"
    text_content += f"Downloads: {downloads:,} | Likes: {likes}\n"
    
    if model_formats:
        text_content += f"Model Formats available: {', '.join(sorted(model_formats))}\n"
    else:
        text_content += "Model Formats available: Unknown or Custom\n"
        
    if quantizations:
        text_content += f"Quantizations Available: {', '.join(sorted(quantizations))}\n"
    else:
        text_content += "Quantizations Available: Standard (No specific low-bit quantization detected. Typical fp16/fp32.)\n"
        
    text_content += f"Important Tags: {', '.join(tags[:10])}...\n"
    
    text_content += f"Sample Model Files: {', '.join(model_files[:5])}"
    if len(model_files) > 5:
        text_content += f" and {len(model_files) - 5} others."
        
    return text_content, repo_id
