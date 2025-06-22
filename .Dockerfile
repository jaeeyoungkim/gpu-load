# Dockerfile

# PyTorch와 CUDA 런타임이 포함된 공식 이미지를 사용합니다.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 1단계에서 만든 5개의 스크립트를 모두 이미지 안으로 복사합니다.
COPY gpu_scenario_*.py ./

# 기본 CMD는 설정하지 않습니다. 쿠버네티스에서 실행할 스크립트를 지정할 것입니다.
