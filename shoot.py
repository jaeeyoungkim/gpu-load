# shoot.py
import torch
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Kubernetes log collection
)
logger = logging.getLogger(__name__)

def run_gpu_load(utilization_level, duration_sec):
    """지정된 시간 동안 목표 사용률에 맞는 GPU 부하를 발생시키는 함수"""

    logger.info(f"Running GPU load at {utilization_level}% for {duration_sec}s.")

    device = torch.device('cuda')
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    work_time = 0.005
    if utilization_level >= 100:
        sleep_time = 0
    else:
        sleep_time = max(0, work_time / (utilization_level / 100.0) - work_time)

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        a = torch.matmul(a, b)
        torch.cuda.synchronize()
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info(f"Completed GPU load at {utilization_level}% for {duration_sec}s.")
    del a, b
    torch.cuda.empty_cache()

def main():
    """5분(300초) 동안 GPU 부하를 단계적으로 증가시키는 메인 함수"""

    logger.info("Starting gradual GPU load test for 5 minutes")

    # 총 실행 시간 (5분 = 300초)
    total_duration = 300

    # 단계별 설정
    # 각 단계는 (사용률, 지속시간) 형태의 튜플로 정의
    stages = [
        (10, 60),    # 1분 동안 10% 사용률
        (30, 60),    # 1분 동안 30% 사용률
        (50, 60),    # 1분 동안 50% 사용률
        (70, 60),    # 1분 동안 70% 사용률
        (90, 60)     # 1분 동안 90% 사용률
    ]

    start_time = time.time()

    # 각 단계별로 부하 실행
    for i, (utilization, duration) in enumerate(stages):
        stage_num = i + 1
        logger.info(f"Stage {stage_num}/5: Starting {utilization}% GPU utilization for {duration} seconds")
        run_gpu_load(utilization, duration)
        logger.info(f"Stage {stage_num}/5: Completed")

    # 총 실행 시간 계산 및 로깅
    total_elapsed = time.time() - start_time
    logger.info(f"Gradual GPU load test completed. Total duration: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in gradual load test: {e}", exc_info=True)
        sys.exit(1)
