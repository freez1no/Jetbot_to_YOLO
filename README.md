<img width="1051" height="527" alt="image" src="https://github.com/user-attachments/assets/3aa88360-2756-4d5d-876e-5177cec76a96" />
<p dir="auto"><a href="https://github.com/freez1no/Omni-Seeker/blob/master/README-KOR.md" rel="nofollow">KOR</a><br></p>
# Nvidia Omniverse Isaac Sim & Lab - Learning to Enhance Object Awareness Using Jetbot's YOLO Model
The project aims to train **Jetbot** robots to autonomously explore and access objects in a **NVIDIA Isaac Lab** simulation environment.

## Introduction
This project establishes a reinforcement learning pipeline that combines 'recognition' and 'action' of robots.
The 'Jetbot.usd' asset is called from the simulation environment (Isaac Lab) to collect the surrounding environment data in real time through the **camera sensor** mounted on the robot, and this vision data is transmitted to the latest object recognition model **YOLO v11** to identify the location of 'people' or pre-defined 'things' as a bounding box.
The reinforcement learning agent utilizes the object recognition results of this YOLO model as Reward and State information to learn an optimal driving policy that approaches the detected object safely and efficiently (properly close).

### Goals
1. Core API based Isaac Lab Code Jetbot, Target prim generation ✓
2. Check Jetbot camera recognition, RL environment, multiple Jetbot and Camera recognition ✓
3. Real-time collection of image data from Jetbot's camera sensors within the Isaac Lab ✓
4. Transform the collected vision data to be recognized by the YOLO v11 model ✓
5. Isaac Lab RL test (working on it)
6. Obtain bounding box information for desired objects (people, objects, etc.) and monitor bounding them in real time
7. Based on object detection information, the reinforcement learning environment is designed to reward the robot when it successfully approaches the target
8. Final Isaac Lab RL, Model Creation
9. Apply and test models to real Jetbot

###  Tech Stack
enviroments : NVIDIA Isaac Lab
Object recognition: YOLO v11
Algorithm: RL Games, skrl, PPO
Data Pipeline: Isaac Sim Camera Sensor
