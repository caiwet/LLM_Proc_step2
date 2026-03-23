from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-4B-Instruct-2507"   # change model here
local_dir = "/data/bwh-comppath-img2/MGH_CID/hf_cache/Qwen3-4B-Instruct-2507"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete:", local_dir)