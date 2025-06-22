import torch, time
print("Scenario 3: Moderately-utilized. Running at ~50% GPU utilization.")
device = torch.device('cuda')
size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
while True:
    a = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.005) # 작업 시간과 동일한 시간 휴식하여 50% 사용률 유지
