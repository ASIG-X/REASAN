rm -rf ./build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
rm -f ../locomotion_node
rm -f ../navigation_node
rm -f ../filter_node
rm -f ../odom_node
rm -f ../ray_estimation_node
cp ./locomotion_node ..
cp ./navigation_node ..
cp ./filter_node ..
cp ./odom_node ..
cp ./ray_estimation_node ..