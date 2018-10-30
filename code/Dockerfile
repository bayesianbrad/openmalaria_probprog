FROM ubuntu:18.04

RUN apt-get update && apt-get install -y xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev cmake

ADD openmalaria-schema-39.0 /code/openmalaria
ADD openmalaria-schema-36.0 /code_old/openmalaria

RUN cd /code/openmalaria && mkdir build && cd build && cmake ..
RUN cd /code/openmalaria/build && make -j4
RUN cd /code/openmalaria/build && ctest -j4

RUN cd /code_old/openmalaria && mkdir build && cd build && cmake ..
RUN cd /code_old/openmalaria/build && make -j4
RUN cd /code_old/openmalaria/build && ctest -j4