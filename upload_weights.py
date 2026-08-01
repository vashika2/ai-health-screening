from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="backend/weights/tb_model_best.pth",
    path_in_repo="tb_model_best.pth",
    repo_id="vashika20/ai-health-screening-weights",
    repo_type="model",
)
print("TB weights uploaded!")

api.upload_file(
    path_or_fileobj="backend/weights/dr_model_best.pth",
    path_in_repo="dr_model_best.pth",
    repo_id="vashika20/ai-health-screening-weights",
    repo_type="model",
)
print("DR weights uploaded!")