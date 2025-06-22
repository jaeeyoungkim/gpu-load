import torch, time
print("Scenario 2: Under-utilized. Running at ~10% GPU utilization.")
device = torch.device('cuda')
size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
while True:
    a = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.045) # 작업 시간의 약 9배 휴식하여 10% 사용률 유지
