This is a simple visualizer for REASAN system state during deployment. It can help debug whether the predicted rays are correct.

Tested on Ubuntu 22.04.

First install dependencies:
```bash
sudo apt install libglfw3-dev libglew-dev libzmq3-dev libtinyxml2-dev
```

Then build:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Then modify the IP address in run_viz.sh. This should be the IP address of e.g. Jetson Orin where REASAN runs. Make sure the computer running this is in the same subnet as the robot and Jetson (basically starting with 192.168).

You can now run_viz.sh to start the visualizer. The robot, the point cloud, the predicted rays should be visualized.