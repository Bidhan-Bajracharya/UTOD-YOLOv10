import torch

print("Hello AI-Simulation Lab")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# cuda version
print("CUDA version:", torch.version.cuda)
# cudnn version
print("CUDNN version:", torch.backends.cudnn.version())
