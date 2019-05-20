FROM ubuntu:18.04

RUN apt-get update && apt-get install -y git xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev cmake python3-pip python3-gdbm libzmq3-dev wget curl tmux nano

RUN mkdir /code

ADD openmalaria /code/openmalaria
RUN cd /code/openmalaria && mkdir build && cd build && cmake ..
RUN cd /code/openmalaria/build && make -j12
#RUN cd /code/openmalaria/build && ctest -j4

