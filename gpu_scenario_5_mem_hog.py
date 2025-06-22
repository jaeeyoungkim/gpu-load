import torch, time
print("Scenario 5: Memory Hog. High memory usage, but 0% GPU utilization.")
device = torch.device('cuda')
tensors = []
# A100 80GB의 약 75% (60GB) 메모리 할당 시도
# 8192x8192 텐서 하나가 약 256MB. 240개면 약 60GB.
print("Allocating a large amount of GPU memory...")
try:
    for i in range(240):
        tensors.append(torch.randn(8192, 8192, device=device))
        print(f"Allocated { (i+1) * 0.25 :.2f} GB...", end='\r')
except RuntimeError:
    print("\nCould not allocate all tensors, but holding what was allocated.")

print("\nMemory allocated. Now sitting idle.")
while True:
    time.sleep(10)
