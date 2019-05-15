FROM ubuntu:18.04

RUN apt-get update && apt-get install -y git xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev cmake python3-pip python3-gdbm libzmq3-dev wget curl tmux nano

RUN mkdir /code
RUN cd /code && git clone --branch v1.10.0 https://github.com/google/flatbuffers.git && cd flatbuffers && cmake -G "Unix Makefiles" && make install
RUN cd /code && git clone --branch 0.4.16 https://github.com/QuantStack/xtl.git && cd xtl && cmake . && make install
RUN cd /code && git clone --branch 0.17.4 https://github.com/QuantStack/xtensor.git && cd xtensor && cmake . && make install
RUN cd /code && git clone --branch 0.3.0  https://github.com/QuantStack/xtensor-io.git && cd xtensor-io && cmake . && make install
RUN cd /code && git clone --branch 0.13.1 https://github.com/QuantStack/xtensor-blas.git && cd xtensor-blas && cmake . && make install

RUN cd /code && git clone --branch master https://github.com/bradleygramhansen/pyprob_cpp.git && cd pyprob_cpp && mkdir build && cd build && cmake ../src && cmake --build . && make install

ADD code/openmalaria /code/openmalaria
RUN cd /code/openmalaria && mkdir build && cd build && cmake ..
RUN cd /code/openmalaria/build && make -j4
#RUN cd /code/openmalaria/build && ctest -j4

RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision jupyter

RUN cd /code && git clone https://github.com/bradleygramhansen/pyprob.git && cd pyprob && pip3 install .
