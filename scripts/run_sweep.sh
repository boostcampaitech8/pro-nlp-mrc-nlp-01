#!/bin/bash

# WandB 로그인 (처음 한 번만)
# wandb login

# Sweep 설정 파일 경로
SWEEP_CONFIG="configs/sweep_config.yaml"

# 프로젝트 이름
PROJECT_NAME="reader-parmeter-tuning"

# Sweep 생성 및 실행
wandb sweep $SWEEP_CONFIG --project $PROJECT_NAME

# 위 명령어가 Sweep ID를 출력하므로, 그 ID를 사용하여 에이전트 실행
# wandb agent <SWEEP_ID> --project $PROJECT_NAME