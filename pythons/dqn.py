from functools import cached_property
import numpy as np
import random
import copy
import datetime
import platform
import torch
from torch.nn.funtional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environmen import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
    import EngineConfigurationChannel

# Goal 관측정보
# DQN 에이전트의 입력으로 사용할 상태의 크기
# 6 채널, 높이 8*8, 너비 84
state_size = [3*2, 64, 84]

# DQN 에이전트의 출력으로 사용할 행동의 크기 (상하좌우)
action_size = 4

# 모델 불러오기 여부
load_model = False

# 모델 학습여부
train_mode = True

# 한번 모델을 학습할때 메모리에서 꺼내는 경험 데이터 수 
batch_size = 32

# 리플레이 메모리의 최대 크기
mem_maxlen = 10000

# 미래에 대한 보상 감가율
discount_factor = 0.9

# 네트워크 학습률
learning_rate = 0.00025

# 학습모드에서 진행할 스텝 수 설정
run_step = 50000 if train_mode else 0

# 평가모드에서 진행할 스텝 수
test_step = 5000

# 학습 시작 전에 리플레이 메모리에 충분한 데이터를 모으기 위해 몇 스텝동안
# 임의의 행동으로 게임 진행할 것인지 설정
train_start_step = 5000

# 타겟 네트워크를 몇 스텝 주기로 업데이트 할지 설정
target_update_step = 500

# 학습 진행상황을 텐서보드에 기록할 주기 설정
print_interval = 10

# 학습 모델을 저장할 에피소드 주기 설정
save_interval = 100

# 평가모드의 앱실론 값
epsilon_eval = 0.05

# 앱실론 초기값
epsilon_init = 1.0 if train_mode else epsilon_eval

# 학습구간의 앱실론 최소값
epsilon_min = 0.1

# 앱실론이 감소되는 구간
explore_step = run_step * 0.8

# 한 스템당 감소하는 앱실론 변화량
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0.

# 시각적 관측 인덱스
VISUAL_OBS = 0

# 목적지 관측 인덱스
GOAL_OBS = 1

# 수치적 관측 인덱스
VECTOR_OBS = 2

# DQN에서는 시각적 관측 인덱스를 사용
OBS = VISUAL_OBS

game = "GridWorld"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/20210514201212"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
