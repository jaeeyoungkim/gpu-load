#!/bin/bash

# --- Whatap 에이전트 설정 ---

# 컨테이너의 작업 디렉터리를 와탭 경로로 설정합니다.
export WHATAP_HOME=${PWD}

# 에이전트 구성에 필수적인 설정 값입니다. (Kubernetes YAML에서 환경 변수로 주입)
# 환경 변수가 설정된 경우에만 config 명령 실행
if [ -n "$whatap_server_host" ] && [ -n "$license" ]; then
    echo "Configuring Whatap agent..."
    whatap-setting-config \
    --host $whatap_server_host \
    --license $license \
    --app_name $app_name \
    --app_process_name $app_process_name
else
    echo "Whatap configuration variables not set. Skipping auto-config."
fi

# --- 애플리케이션 실행 ---

# "$@"는 Dockerfile의 CMD나 Kubernetes의 command/args로 전달된 모든 인자를 의미합니다.
# exec whatap-start-agent "$@"는
# "whatap-start-agent [CMD 또는 command로 전달된 명령어]" 형태로 실행됩니다.
# 이를 통해 어떤 명령이 들어오든 whatap-start-agent로 감쌀 수 있습니다.
echo "Starting application with whatap-start-agent wrapper..."
exec whatap-start-agent "$@"
