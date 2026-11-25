
import modal

app = modal.App("wan-video-lab")

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def gpu_info():
    import torch

    print("GPU environment check")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

@app.local_entrypoint()
def main():
    gpu_info.remote()

