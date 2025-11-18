<img width="1051" height="527" alt="image" src="https://github.com/user-attachments/assets/3aa88360-2756-4d5d-876e-5177cec76a96" />
<p dir="auto"><a href="https://github.com/freez1no/Omni-Seeker/blob/master/README.md" rel="nofollow">ENG</a><br></p>

# Nvidia Omniverse Isaac Sim & Lab - Jetbot의 YOLO 모델을 이용한 객체 인식 강화학습 주행
이 프로젝트는 **NVIDIA Isaac Lab** 시뮬레이션 환경에서 **Jetbot** 로봇이 자율적으로 객체를 탐색하고 접근하도록 훈련하는 것을 목표로 한다.

## 소개
본 프로젝트는 로봇의 '인식'과 '행동'을 결합하는 강화학습 파이프라인을 구축한다.
시뮬레이션 환경(Isaac Lab)에서 `Jetbot.usd` 에셋을 불러와 로봇에 탑재된 **카메라 센서**를 통해 실시간으로 주변 환경 데이터를 수집하고, 이 비전 데이터는 최신 객체 인식 모델인 **YOLO v11**로 전송되어 '사람' 또는 사전에 정의된 '사물'의 위치를 바운딩박스로 식별한다.
강화학습 에이전트는 이 YOLO 모델의 객체 인식 결과를 Reward 및 State 정보로 활용하여, 탐지된 객체에게 안전하고 효율적으로 접근하는(적당히 가까이 다가가는) 최적의 주행 policy를 학습한다.

### 목표
1. Core API 기반 Isaac Lab Code로 Jetbot, Target prim 생성 ✓
2. Jetbot 카메라 인식 확인, RL 환경에서, 여러대의 Jetbot 및 Camera 인식 확인 ✓
3. Isaac Lab 내 Jetbot의 카메라 센서로부터 이미지 데이터를 실시간으로 수집 ✓
4. 수집된 비전 데이터를 YOLO v11 모델이 인식하도록 변환 ✓
5. Isaac Lab RL 테스트 (working on it)
6. 원하는 객체(사람, 사물 등)의 바운딩 박스 정보를 획득, 바운딩 박스 실시간 모니터링
7. 객체 탐지 정보를 기반으로, 로봇이 목표물에 성공적으로 접근했을 때 보상을 제공하는 강화학습 환경 설계
8. 최종 Isaac Lab RL, 모델 생성
9. 실제 Jetbot에 모델 적용 및 테스트

###  기술 스택
환경 : NVIDIA Isaac Lab
객체인식 : YOLO v11
알고리즘 : RL Games, skrl, PPO
데이터 파이프라인 : Isaac Sim Camera Sensor
