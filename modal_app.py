
import modal

app = modal.App("modal-wan-video-lab")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(image=image, gpu="T4", timeout=300)
def inspect_gpu():
    import torch

    result = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "No GPU detected"
        ),
    }

    print(result)
    return result

@app.local_entrypoint()
def main():
    inspect_gpu.remote()

