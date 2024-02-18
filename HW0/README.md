# HW0
## Introduction
In the following homeworks and final project we will need to use the CARLA simulator to simulate various traffic scenes. CARLA supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, and most importantly, a PythonAPI for users to easily customize the environment and get control of the vehicles. 

This homework will guide you through the installation of CARLA and provide 4 driving benchmarks and 3 different baseline driving models to give you a quick view on how autonomous driving systems looked like in CARLA.

## System Requirements
- Unbuntu 18.04 or 20.04
- 8GB GPU or even better
- At least 50GB free space

## Setting up your environment
In this homework we will use 2 versions of CARLA, 0.9.10 and 0.9.14, the version depending on which benchmark you choose. This section will guide you through the installation steps of both versions, and the corresponding version of each benchmark will be specified  in the Benchmark section.

### CARLA 0.9.10
  1. Download the packaged CARLA realse  
      ```shell
      chmod +x setup_carla_10.sh
      ./setup_carla_10.sh
      ```
  2. Open CARLA
      ```shell
      cd carla_10
      ./CarlaUE4.sh
      ``` 
      If you see a screen like this, it means CARLA 0.9.10 is successfully installed.
      [](./assests/carla_10_example.png)

### CARLA 0.9.14
  1. Download the packaged [CARLA release](https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz).
  2. Unzip the package into a folder, e.g. carla_14
  3. Open CARLA
      ```shell
      cd carla_14
      ./CarlaUE4.sh
      ``` 
      If you see a screen like this, it means CARLA 0.9.14 is successfully installed.
      [](./assests/carla_14_example.png)    
