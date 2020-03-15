mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
mv ./*.so ../

#rm -rf build