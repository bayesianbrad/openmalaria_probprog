FROM ubuntu:18.04

RUN apt-get update && apt-get install -y git xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev cmake python3-pip python3-gdbm libzmq3-dev wget curl tmux nano

RUN mkdir /code
RUN cd /code && git clone --branch v1.10.0 https://github.com/google/flatbuffers.git && cd flatbuffers && cmake -G "Unix Makefiles" && make install
RUN cd /code && git clone --branch 0.4.16 https://github.com/QuantStack/xtl.git && cd xtl && cmake . && make install
RUN cd /code && git clone --branch 0.17.4 https://github.com/QuantStack/xtensor.git && cd xtensor && cmake . && make install
RUN cd /code && git clone --branch 0.3.0  https://github.com/QuantStack/xtensor-io.git && cd xtensor-io && cmake . && make install
RUN cd /code && git clone --branch 0.13.1 https://github.com/QuantStack/xtensor-blas.git && cd xtensor-blas && cmake . && make install

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install -y anaconda jupyter

RUN cd /code && git clone --branch master https://github.com/probprog/pyprob_cpp.git && cd pyprob_cpp && mkdir build && cd build && cmake ../src && cmake --build . && make install
RUN cd /code && git clone https://github.com/probprog/pyprob.git && cd pyprob && pip3 install .


ADD openmalaria_pp /code/openmalaria_pp
RUN cd /code/openmalaria_pp && mkdir build && cd build && cmake ..
RUN cd /code/openmalaria_pp/build && make -j4
# RUN cd /code/openmalaria_pp/build && ctest -j4



