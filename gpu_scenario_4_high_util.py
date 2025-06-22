import torch, time
print("Scenario 4: Well-utilized. Running at ~90% GPU utilization.")
device = torch.device('cuda')
size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
while True:
    a = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.00055) # 작업 시간의 1/9만 휴식하여 90% 사용률 유지
