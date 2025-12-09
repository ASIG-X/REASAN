# REASAN: Learning Reactive Safe Navigation for Legged Robots

REASAN is a safe reactive navigation system for the Unitree Go2 robot, featuring modularized system design and real-time on-board deployment.

Feel free to open issues if you encounter any problems or have any questions when using the code!

## Installation

First create a new conda environment:
```
conda create -n env_reasan python=3.10
conda activate env_reasan
```

Install PyTorch:
```
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Install IsaacSim 4.5.0:
```
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

Install IsaacLab inside the cloned repo:
```
cd training
cd IsaacLab
./isaaclab.sh --install
```

Install rsl_rl:
```
cd ..
pip install -e ./rsl_rl
```

Install REASAN:
```
pip install -e ./go2_lidar
pip install -e ./ray_predictor
```

## Play Trained Models in IsaacLab

We provide pretrained checkpoints for the locomotion, safety shield (denoted as "filter" in the code), navigation, as well as exteroceptive estimation (denoted as "ray predictor" or "ray estimator" in the code) module.
Note that the checkpoint files are forcefully added to git, which should be ignored by the gitignore file.
If you get warnings about this in vscode, it's not a problem.

To play the pretrained locomotion policy:
```
# for playing and training, we assume the current working directory in the terminal is 'training'
python scripts/play_loco.py --load_run loco_1
```

To play the pretrained safety shield policy:
```
python scripts/play_filter.py --load_run filter_1 --with_dyn_obst --confirm
# add --keyboard for keyboard control: WSAD (vx and vy) + CV (yaw rate)
```

To play the pretrained navigation policy:
```
python scripts/play_nav.py --load_run nav_1 --confirm --with_dyn_obst
```

## Training

Train locomotion policy:
```
# first you need to login with wandb in your terminal: https://docs.wandb.ai/models/ref/cli/wandb-login
# first stage training:
python scripts/train_loco.py --run_name loco_new --num_envs 4096 --max_iterations 5000 --wandb_proj go2_loco
# second stage training for higher speed and better robustness:
python scripts/train_loco.py --run_name loco_new --resume --load_run loco_new --num_envs 4096 --max_iterations 5000 --wandb_proj go2_loco --second_stage
```

Train safety shield policy:
```
# first stage training:
python scripts/train_filter.py --run_name filter_new --num_envs 1024 --max_iterations 10000 --wandb_proj go2_filter --confirm
# second stage training
python scripts/train_filter.py --run_name filter_new --resume --load_run filter_new --num_envs 1024 --max_iterations 10000 --wandb_proj go2_filter --with_dyn_obst --confirm
```

Train navigation policy:
```
python scripts/train_nav.py --run_name nav_new --num_envs 1024 --max_iterations 10000 --wandb_proj go2_nav --with_dyn_obst --confirm
```

The training will generate runs and checkpoins inside the `training/logs/rsl_rl/go2_lidar` folder.
You can play the trained models with the play scripts.

## Exteroceptive Estimator

Inside `training/ray_predictor` is the code for training the exteroceptive estimator.

First, you need to collect data with the safety shield policy:
```
# create data folder
mkdir ray_predictor/ray_predictor/data

# collect training data
python scripts/play_filter.py --load_run filter_1 --with_dyn_obst --confirm --collect train --headless --num_envs 256

# collect validation data
python scripts/play_filter.py --load_run filter_1 --with_dyn_obst --confirm --collect val --headless --num_envs 256
```
The datasets will be saved inside the `training/ray_predictor/ray_predictor/data` folder.

You can visualize the data with:
```
cd ray_predictor/ray_predictor

pip install plotly nicegui
python visualize_ray_estimator_data.py --input data/train.h5
```
This will open a webpage in your browser with the visualization (using NiceGUI).

Next, start the training:
```
python train_ray_estimator.py --mode train --h5_file data/train.h5 --val_h5_file data/val.h5 --history_frames 15 --batch_size 256 --num_epochs 100 --learning_rate 1e-4 --device cuda
```

You can inference with the model and visualize the results:
```
# run inference
python train_ray_estimator.py --mode predict --h5_file data/train.h5 --output_file data/pred.h5 --checkpoint ray_predictor_best.pth --history_frames 15

# run visualization
python visualize_ray_estimator_data.py --input data/train.h5 --pred data/pred.h5
```

Finally, export the model to onnx format, which can be used for on-board deployment:
```
python train_ray_estimator.py --mode export --checkpoint ray_predictor_best.pth --output_file ray_predictor_new.onnx --export_format onnx --history_frames 15
```

## Real Robot Deployment

Please check [README](deployment/README.md) in the deployment folder.

# License

The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

# Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

**Safety Warning:**
This code is intended for research and educational purposes only. Deploying code on physical robots involves significant risks, including but not limited to hardware damage, property damage, and personal injury.
1. **Hardware Safety:** The control policies and algorithms provided may generate commands that exceed the physical limits of the robot hardware, potentially leading to overheating, mechanical failure, or permanent damage.
2. **Operational Environment:** Ensure the robot is operated in a safe, controlled environment with adequate space and safety barriers. Do not operate the robot near people, pets, or fragile objects.
3. **Emergency Stop:** Always have a reliable, hardware-based emergency stop (E-Stop) mechanism immediately accessible when running the robot. Software-based stops may fail.
4. **Supervision:** Never leave the robot unattended while it is powered on or operating.

By using this software, you acknowledge and agree that you are solely responsible for ensuring the safety of the deployment and for any consequences resulting from its use. The authors assume no responsibility for any damage to equipment, injury to persons, or other losses caused by the use of this code.

**The authors strictly oppose any military use of this work!**