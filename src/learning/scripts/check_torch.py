import torch

print("PyTorch 설치됨:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("사용 가능한 GPU 이름:", torch.cuda.get_device_name(0))
else:
    print("CUDA를 사용할 수 없습니다.")
