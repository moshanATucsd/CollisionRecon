
ceres: ceres.cc
	clang++ -std=c++11 -g -O0 -shared -o ceres.so ceres.cc -I /usr/include/eigen3 -I /usr/local/include -I ./pybind11/include/ -lgomp -lglog -lceres -lcholmod -fPIC -L /usr/local/lib -L /usr/lib/x86_64-linux-gnu/ -L ~/anaconda3/lib $(shell python3-config --libs --cflags)
