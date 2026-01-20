# main.py (v4 - CuPy Version)
from fastapi import FastAPI, BackgroundTasks
from typing import Dict
import time
import logging
import cupy as cp

# --- 표준 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="GPU Load Test API (CuPy Edition)")

# --- GPU 부하를 주는 핵심 함수 (CuPy) ---
def get_working_device_id():
    """사용 가능한 첫 번째 CUDA 장치 ID를 찾아서 반환합니다."""
    try:
        count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"Checking {count} visible CUDA devices (CuPy)...")
        for i in range(count):
            try:
                with cp.cuda.Device(i):
                    # 작은 배열을 할당하여 접근 권한 확인
                    cp.array([1.0])
                logger.info(f"Device {i} is accessible.")
                return i
            except Exception as e:
                logger.warning(f"Device {i} is not accessible: {e}")
        raise RuntimeError("No accessible CUDA devices found among visible devices.")
    except Exception as e:
        # 드라이버 자체가 없는 경우 등
        logger.error(f"Failed to query CUDA devices: {e}")
        raise

def run_gpu_load(utilization_level: int, duration_sec: int):
    """지정된 시간 동안 목표 사용률에 맞는 GPU 부하를 발생시키는 함수 (CuPy 기반)"""
    
    # 기본 가용성 체크
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            logger.error("No CUDA devices found.")
            return
    except Exception as e:
        logger.error(f"CUDA driver error: {e}")
        return

    try:
        device_id = get_working_device_id()
    except Exception as e:
        logger.error(f"Failed to find a working GPU: {e}")
        return

    logger.info(f"STARTING GPU load task: {utilization_level}% for {duration_sec}s using device {device_id}")

    # 해당 장치 컨텍스트 진입
    with cp.cuda.Device(device_id):
        size = 8192  # 행렬 크기 (조정 가능)
        try:
            # 랜덤 행렬 생성
            a = cp.random.randn(size, size, dtype=cp.float32)
            b = cp.random.randn(size, size, dtype=cp.float32)
        except Exception as e:
            logger.error(f"Failed to initialize tensors on device {device_id}: {e}")
            return

        work_time = 0.005 # 작업 단위 시간 (초)
        
        # Sleep 시간 계산
        if utilization_level >= 100:
            sleep_time = 0
        else:
            # (Work / Util) - Work = Sleep
            # 예: 0.005 / 0.5 - 0.005 = 0.005 (50% Load)
            sleep_time = max(0, work_time / (utilization_level / 100.0) - work_time)

        start_time = time.time()
        
        # 부하 루프
        while time.time() - start_time < duration_sec:
            # 행렬 곱셈 (실제 연산)
            c = cp.matmul(a, b)
            
            # 동기화 (연산이 끝날 때까지 대기해야 부하가 유지됨)
            cp.cuda.Stream.null.synchronize()
            
            # 휴식 (사용률 조절)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"FINISHED GPU load task: {utilization_level}% for {duration_sec}s.")
        
        # 메모리 정리
        del a, b, c
        cp.get_default_memory_pool().free_all_blocks()

# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root() -> Dict[str, str]:
    logger.info("Root endpoint '/' was called.")
    return {"message": "GPU Load Test API (CuPy) is running. Use /load/{level} or /latency/{level}."}

@app.get("/load/{level}")
def trigger_load(level: int, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """[비동기] GPU 부하 작업을 백그라운드에서 시작시키고 즉시 응답합니다."""
    if not (0 < level <= 100):
        return {"error": "Please provide a utilization level between 1 and 100."}
    
    duration = 30
    import random
    random_sleep = random.uniform(1.1, 10.0)
    time.sleep(random_sleep)
    logger.info(f"'/load/{level}' called. Scheduling a {duration}s task in the background.")
    
    background_tasks.add_task(run_gpu_load, level, duration)
    
    return {"message": f"GPU load test accepted. Will run in the background at ~{level}% for {duration}s."}

@app.get("/latency/{level}")
def trigger_latency_load(level: int) -> Dict[str, str]:
    """[동기] GPU 부하 작업을 '동기적으로' 실행하여 긴 Latency를 시뮬레이션합니다."""
    if not (0 < level <= 100):
        return {"error": "Please provide a utilization level between 1 and 100."}
        
    duration = 30 
    logger.info(f"'/latency/{level}' called. Running a {duration}s task synchronously (blocking).")

    run_gpu_load(level, duration)
    
    logger.info(f"Synchronous task for '/latency/{level}' finished. Sending response.")
    return {"message": f"Synchronous GPU load test finished after {duration} seconds at ~{level}%."}
