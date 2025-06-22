# Dockerfile (v2 - with Whatap Entrypoint)

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime
WORKDIR /app

# --- [추가] Whatap 에이전트 설치 ---
# 이 부분은 와탭에서 제공하는 설치 방법에 따라 추가되어야 합니다.
# 예를 들어, RUN wget ... && tar -xf ... 와 같은 명령이 될 수 있습니다.
# 지금은 에이전트가 베이스 이미지에 이미 설치되어 있다고 가정합니다.
# --------------------------------

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# --- [변경점 1] entrypoint.sh 복사 및 실행 권한 부여 ---
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# --- [변경점 2] 컨테이너의 진입점으로 entrypoint.sh 설정 ---
ENTRYPOINT ["/app/entrypoint.sh"]

# --- [변경점 3] 기본 명령어(CMD)의 오타 수정 및 역할 명시 ---
# 이 CMD는 ENTRYPOINT 스크립트의 "$@" 인자로 전달됩니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
