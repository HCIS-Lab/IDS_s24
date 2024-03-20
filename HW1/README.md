# HW1
In this homework you will learn to perform basic occupancy forecasting from ground truth bounding boxes with a constant velocity model and a learning-based model. You need to implement some critical functions used in data preprocessing, validation, and visualization.

# Environment
You can use the environment of HW0 directly

# Usage
```shell
bash run.sh $MODEL $FORECAST_TIME $VALIDATE
# e.g. bash run.sh learn 0.5 0
# $MODEL == constant setting is automatically run on validation set.
```

> **:heavy_exclamation_mark:**
> If you want to run learning-based model on validation set only, please remember to specify --load_file in [run.sh](./run.sh) to the file path of your model checkpoint. 