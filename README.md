## Tracking freely behaving specimens in real time

PyTorch package to enable real-time tracking of freely behaving animals with high precision and ultralow latency (< 4 ms/frame, > 250 Hz), using generative models, prior-informed position estimation, inference runtime optimization, and a rapid hyperparameter search strategy. Prior-informed position estimation reduces the tracking error to less than 1 in a million frames. Inference runtime optimization reduces inference runtime to < 2ms per frame, and hyperparameter search optimization enables a 30-fold increase in search speed, leading to 60,000 frames per second search speed. 

<img src="./tracking.svg" width=100% height=100%>

a) Trace of larval zebrafish (danio rerio) in behavioral arena over the course of a 1.68 million frame experiment (112 minutes).\
b) The specimen freely moves in the arena while being tracked. Relevant behavioral data can accurately be recorded in real time with high temporal resolution, enabling the study of corresponding neural activity during unconstrained behavior.

Pull with

`git clone https://github.com/sebvoigtlaender/real_time_tracking.git`

## Dependencies

Python 3.6 or later. Other dependencies can be installed via

`pip install -r requirements.txt`

To use inference runtime optimization, also run

`pip install nvidia-tensorrt`

## Basic usage

Run the `basic_tutorial.ipynb` (make sure jupyter is up to date with `pip install -U jupyter`)

To train in the console, run

`python train.py`

For full control over the parameters, specify the arguments, e.g.,

`python train.py --len-dataset=1000 --lr=0.003 --n-episodes=10000`

For evaluation, hyperparameter search, and testing, use

`evaluate.py`\
`search_hp.py`\
`test.py`

## Advanced usage

To understand the usage more in-depth, run and understand the `advanced_tutorial.ipynb`. Furthermore, read `arguments.py` and `config.py`.\
To customize the tracker to your needs, the user might need to change the `config.py`, `data_utils.pt`, `file_sys_utils.py`, and `tracking_utils.py`. The docstrings should help in the process.

## Inference acceleration

Running the model with 16 floating point precision at test time significantly speeds up inference.
If you have `tensorrt` installed, simply run

`python tensorrt_test.py`

## Test tensorrt in docker container

If you cannot install `tensorrt` on your server, you can try it first in a docker container.
Install docker and run

`docker run --gpus all -it nvcr.io/nvidia/pytorch:22.01-py3`

If you get the error:

`Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

run: 

```
sudo apt-get install -y docker nvidia-container-toolkit
systemctl restart dockerd
```
Check if the container is there with

`docker ps -a`

You will find the container id in the displayed list. Start and enter container by running: 

```
docker start container_id
docker attach container_id
```

You are in the `/workspace` directory of the container. We recommend creating a new folder, e.g. `dasher`.

```
mkdir dasher
cd dasher
```

Now you can import the `.py` files in the container by either leaving the container with `exit` or by opening another terminal and using the command

`docker cp FILE.py CONTAINER_ID:/workspace/dasher`

where `FILE` is the filename, e.g., `cross_validate`.

In the container, you will find the file in the `dasher` folder.\
To get a unittest up and running, you can simply `docker cp` the files `tensorrt_fpn.py`, `tensorrt_resnet.py`, `tensorrt_model.py`, and `tensorrt_unittest.py`, with

```
docker cp tensorrt/tensorrt_fpn.py CONTAINER_ID:/workspace/dasher/
docker cp tensorrt/tensorrt_resnet.py CONTAINER_ID:/workspace/dasher/
docker cp tensorrt/tensorrt_model.py CONTAINER_ID:/workspace/dasher/
docker cp tensorrt/tensorrt_unittest.py CONTAINER_ID:/workspace/dasher/
```

and run

`python tensorrt_unittest.py`

inside the container.