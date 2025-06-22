# gpu_scenario_5_mem_hog.py (5분 실행 버전)
import torch
import time

print("Scenario 5: Memory Hog. High memory usage, but 0% GPU utilization.")
print("This job will run for 5 minutes and then exit automatically.")

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

print("\nMemory allocated. Now sitting idle for 5 minutes.")

# --- [수정됨] 5분(300초) 타이머 ---
start_time = time.time()
while time.time() - start_time < 300:
    # 10초마다 점검 메시지 출력 (선택사항)
    time.sleep(10)
    remaining_time = 300 - (time.time() - start_time)
    print(f"Holding memory... Time remaining: {remaining_time:.0f} seconds.", end='\r')

# ---------------------------------

print("\n5-minute run finished. Releasing memory.")
del tensors
torch.cuda.empty_cache()
print("Script finished.")

