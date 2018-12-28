FROM ubuntu:18.04 

RUN apt-get update && apt-get install -y git xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev cmake python3-pip python3-gdbm libzmq3-dev wget curl tmux nano

ADD openmalaria-schema-39.0 /code/openmalariapp
RUN cd /code/openmalariapp && mkdir build && cd build && cmake ..
RUN cd /code/openmalariapp/build && make -j4
# RUN cd /code/openmalaria_schema_pp/build && ctest -j4
