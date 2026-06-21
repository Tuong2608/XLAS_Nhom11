from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="links-ads/wildfires-cems",
    repo_type="dataset",
    local_dir="./wildfires-cems",
    local_dir_use_symlinks=False,
)
