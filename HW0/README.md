# HW0
## Introduction
In the following homeworks and final project we will need to use the CARLA simulator to simulate various traffic scenes. CARLA supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, and most importantly, a PythonAPI for users to easily customize the environment and get control of the vehicles. 

This homework will guide you through the installation of CARLA and provide 4 driving benchmarks and 3 different baseline driving models to give you a quick view on how autonomous driving systems looked like in CARLA.

We expect you to start to investigate the topic of your midterm presentation and final project based on your observation after running some of the baselines on one or more benchmarks we provided.

## System Requirements
- Unbuntu 18.04 or 20.04
- Intel i7 gen 9th - 14th / Intel i9 gen 9th - 14th / AMD ryzen 7 / AMD ryzen 9
- 16 GB RAM memory
- 8GB GPU or even better, e.g.:
  - NVIDIA RTX 2070 / NVIDIA RTX 2080 / NVIDIA RTX 3070, NVIDIA RTX 3080 
- At least 50GB free space (not including the space for training data in future homeworks or your project)

## Setting up your environment
In this homework we will use 2 versions of CARLA, 0.9.10 and 0.9.14, the version depending on which benchmark you choose. This section will guide you through the installation steps of both versions, and the corresponding version of each benchmark will be specified  in the [Benchmark](#benchmark) section.

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
      ![](./assets/carla_10_example.png)
  3. Run manual_control, open another terminal
      ```shell
      cd carla_10/PythonAPI/examples
      python manual_control.py
      ``` 

### CARLA 0.9.14
  1. Download the packaged [CARLA release](https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz).
  2. Unzip the package into a folder, e.g. carla_14
  3. Open CARLA
      ```shell
      cd carla_14
      ./CarlaUE4.sh
      ``` 
      If you see a screen like this, it means CARLA 0.9.14 is successfully installed.
      ![](./assets/carla_14_example.png)

### Create conda environment
```shell
conda env create -f environment.yml
```

### Manual control
  open another terminal
  ```shell
  cd carla_14/PythonAPI/examples
  python manual_control.py
  ``` 

## Benchmark
### Longest6
1. Start a CARLA 0.9.10 server
    ```shell
    cd carla_10
    ./CarlaUE4.sh
    ```
2. Open another terminal
    ```shell
    cd HW0
    bash longest6/scripts/run_evaluation.sh ${CARLA_ROOT} ${HOMEWORKD_ROOT} ${MODEL_NAME}
    # e.g. bash longest6/scripts/run_evaluation.sh ./carla_10 . TFPP
    ```


[Paper](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf), [GitHub](https://github.com/autonomousvision/transfuser?tab=readme-ov-file)

### DOS
1. Start a CARLA 0.9.10 server
    ```shell
    cd carla_10
    ./CarlaUE4.sh
    ```
2. Open another terminal
    ```shell
    cd HW0
    bash DOS/leaderboard/scripts/run_evaluation.sh ${CARLA_ROOT} ${HOMEWORKD_ROOT} ${MODEL_NAME}
    # e.g. bash longest6/scripts/local_evaluation.sh ./carla_10 . TCP
    ```
3. Update [DOS/leaderboard/scripts/run_evaluation.sh](./DOS/leaderboard/scripts/run_evaluation.sh) by setting `ROUTES` to `DOS_benchmark/DOS_0X_town05.xml`, `SCENARIOS` to `DOS_benchmark/DOS_0X_town05.json` (where X is the scenario id), and ensuring that `TEAM_CONFIG`, `TEAM_AGENT`, and `CARLA_ROOT` are correctly configured.

    [Paper](https://arxiv.org/pdf/2305.10507.pdf), [GitHub](https://github.com/opendilab/DOS)



### CARLA leaderboard 2.0
1. Start a CARLA 0.9.14 server
    ```shell
    cd carla_14
    ./CarlaUE4.sh
    ```
2. Open another terminal
    ```shell
    cd HW0
    bash leaderboard_2.0/scripts/run_evaluation.sh ${CARLA_ROOT} ${HOMEWORKD_ROOT} ${MODEL_NAME}
    # e.g. bash leaderboard_2.0/scripts/run_evaluation.sh ./carla_14 . TFPP
  [CARLA leaderboard 2.0](https://leaderboard.carla.org/get_started/)

## Baseline
### Transfuser++
  Change ${MODEL_NAME} to `TFPP` ot run Transfuser++ model, e.g.:
  ```shell
  bash leaderboard_2.0/scripts/run_evaluation.sh ./carla_14 . TFPP
  ```
  
  [Paper](https://arxiv.org/pdf/2306.07957.pdf), [GitHub](https://github.com/autonomousvision/carla_garage)

### TCP
  Change ${MODEL_NAME} to `TCP` ot run TCP model, e.g.:
  ```shell
  bash leaderboard_2.0/scripts/run_evaluation.sh ./carla_14 . TCP
  ```
  [Paper](https://arxiv.org/pdf/2206.08129.pdf), [GitHub](https://github.com/OpenDriveLab/TCP), [Model_ckpt](https://hkustconnect-my.sharepoint.com/personal/qzhangcb_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fqzhangcb%5Fconnect%5Fust%5Fhk%2FDocuments%2FPublic%5FShared%5FOnline%2FPre%2Dtrain%20weights%2FTPC%5FTrained%5FModel%2Fbest%5Fmodel%2Eckpt&parent=%2Fpersonal%2Fqzhangcb%5Fconnect%5Fust%5Fhk%2FDocuments%2FPublic%5FShared%5FOnline%2FPre%2Dtrain%20weights%2FTPC%5FTrained%5FModel&ga=1)

### PlanT
  Change ${MODEL_NAME} to `PlanT` ot run PlanT model, e.g.:
  ```shell
  bash leaderboard_2.0/scripts/run_evaluation.sh ./carla_14 . PlanT
  ```
  > **:heavy_exclamation_mark:**
  > Remember to set `TRACK` to `MAP` before running PlanT
  
  [Paper](https://arxiv.org/pdf/2210.14222.pdf), [GitHub](https://github.com/autonomousvision/plant)

## Submission
Please submit a 1 minute video of you running mannul_control.py on 1 version of CARLA before 3/6 23:59, this sumbission accounts for 2% of your final score. 