# main.py (v3 - Final Version with Logging and Latency Endpoint)
from fastapi import FastAPI, BackgroundTasks
from typing import Dict, Optional
import torch
import time
import logging
import random

# --- [개선점 1] 표준 로깅 설정 ---
# 로그 포맷과 레벨을 지정하여, kubectl logs에서 더 유용한 정보를 볼 수 있게 합니다.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="GPU Load Test API")


# --- GPU 부하를 주는 핵심 함수 ---
def run_gpu_load(utilization_level: int, duration_sec: int):
    """지정된 시간 동안 목표 사용률에 맞는 GPU 부하를 발생시키는 함수"""

    logger.info(f"STARTING GPU load task: {utilization_level}% for {duration_sec}s.")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Skipping GPU load.")
        return

    # 명시적으로 cuda:0 사용 (Kubernetes에서 단일 GPU가 할당된 경우 항상 인덱스 0)
    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.error("No CUDA devices found. Skipping GPU load.")
        return
    
    device_index = 0  # 항상 첫 번째 사용 가능한 GPU 사용
    device = torch.device(f'cuda:{device_index}')
    logger.info(f"Using GPU device {device_index}: {torch.cuda.get_device_name(device_index)} (total devices: {device_count})")
    
    size = 8192
    try:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
    except Exception as e:
        logger.error(f"Failed to allocate tensors: {e}")
        return

    # Estimated work time for one matmul iteration.
    # Adjust this if the GPU is significantly faster/slower.
    work_time = 0.005

    if utilization_level >= 100:
        sleep_time = 0.0
    else:
        # Duty Cycle Formula derived from: Work / (Work + Sleep) = Util / 100
        # => Sleep = Work / (Util/100) - Work
        sleep_time = max(0.0, work_time / (utilization_level / 100.0) - work_time)

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        # Compute
        a = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Sleep to adjust load
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info(f"FINISHED GPU load task: {utilization_level}% for {duration_sec}s.")
    del a, b
    torch.cuda.empty_cache()


# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root() -> Dict[str, str]:
    logger.info("Root endpoint '/' was called.")
    return {"message": "GPU Load Test API is running. Use /load/{level} or /latency/{level}."}


@app.get("/load/{level}")
def trigger_load(level: int, background_tasks: BackgroundTasks, duration: int = 30) -> Dict[str, str]:
    """[비동기] GPU 부하 작업을 백그라운드에서 시작시키고 즉시 응답합니다."""
    if not (0 < level <= 100):
        return {"error": "Please provide a utilization level between 1 and 100."}

    # Random sleep to simulate realistic/jittery arrival or startup delay
    random_sleep = random.uniform(1.1, 10.0)
    time.sleep(random_sleep)

    logger.info(f"'/load/{level}' called. Scheduling a {duration}s task in the background.")

    # FastAPI의 BackgroundTasks를 사용하여 즉시 응답하고, 작업은 백그라운드에서 실행
    background_tasks.add_task(run_gpu_load, level, duration)

    return {"message": f"GPU load test accepted. Will run in the background at ~{level}% for {duration}s."}


# --- [개선점 2] Latency 시나리오를 위한 엔드포인트 ---
@app.get("/latency/{level}")
def trigger_latency_load(level: int, duration: int = 30) -> Dict[str, str]:
    """[동기] GPU 부하 작업을 '동기적으로' 실행하여 긴 Latency를 시뮬레이션합니다."""
    if not (0 < level <= 100):
        return {"error": "Please provide a utilization level between 1 and 100."}

    logger.info(f"'/latency/{level}' called. Running a {duration}s task synchronously (blocking).")

    # 여기서는 background_tasks를 사용하지 않고 직접 함수를 호출합니다.
    # 따라서 이 요청은 duration(초) 후에 응답이 완료됩니다.
    run_gpu_load(level, duration)

    logger.info(f"Synchronous task for '/latency/{level}' finished. Sending response.")
    return {"message": f"Synchronous GPU load test finished after {duration} seconds at ~{level}%."}


@app.get("/gpu-info")
def gpu_info():
    """GPU 정보를 반환합니다."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            devices.append(torch.cuda.get_device_name(i))
        return {
            "available": True,
            "count": device_count,
            "devices": devices,
            "current_device": torch.cuda.current_device(),
        }
    else:
        return {
            "available": False,
            "message": "No GPU detected",
        }
