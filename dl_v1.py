from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="gy65896/FVessel", repo_type="dataset",
                      allow_patterns=["FVessel_V1.0.zip"],
                      local_dir="data/raw/fvessel_dist")
print("FVessel V1.0 ->", p)
