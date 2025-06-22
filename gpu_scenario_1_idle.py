import torch, time
print("Scenario 1: Idle Hog. Holding GPU with 0% utilization. (To stop: Ctrl+C)")
device = torch.device('cuda')
# GPU를 점유하기 위해 작은 텐서 하나만 할당
dummy_tensor = torch.randn(1, 1, device=device)
while True:
    time.sleep(10) # 거의 아무것도 하지 않고 계속 대기
