todo
====

Add notes to all the .cpp files that we modify, to state exactly
what we modified.



openmalaria dependencies (Ubuntu):
- apt packages: xsdcxx libxerces-c-dev libgsl-dev libboost-all-dev


How to build wiki locally:

First download Gollum:

On mac:

- sudo gem install gollum

Input/Outputs to the model
==========================

What can we designate as input/output:
  - Inputs
     - Mosquito nets
     - Vaccination, type of vaccination
     - Prophylactic
  - Outputs
     - "Survey measures"
       - nHost
       - nPatent
     - Mortality rate
     - Probability of seeking medical help



How to build Docker image:
==========================

- docker build . -t openmalariapp
- docker run --rm -it (for interactive usage, will remove the container from memory) (-it interactive attach to terminal)

To attach a local drive / folder use
<host file system>:<inside docker>
- docker run --rm -it -v $PWD:/home bradleygh/openmalariapp
Connecting docker to the external file system:


How to run Jupyter inside Docker:

For linux

- docker run --rm -it -v $PWD:/home --net=host bradleygh/openmalariapp
run the following inside the container:  jupyter notebook --allow-root


For Mac

docker run --rm -it -p 127.0.0.1:8889:8889 -v $PWD:/home gbaydin/openmalariapp jupyter notebook --port 8889 --allow-root --ip=0.0.0.0


Creating an experiment
======================

Create a  directory in your local machine, for example called examples
- cd /home
- mkdir examples
- cd examples

within the folder add the following files:
- scenario_current.xsd
- <name_of_the_scenario_you_want_to_run>.xml

Add the openmalaria executable to the folder to, i.e
- cp /code/openmalaria/build/openMalaria examples/openMalaria

- <add_any_input_csv_or_txt_files>


Running the simulator once an experiment has been created
=========================================================

- cd ./examples
- docker run --rm -it -v $PWD:/home/examples bradleygh/openmalariapp
- cd /home/examples/
- ./openMalaria -s <name_of_the_scenario_you_want_to_run>.xml


Debugging by modifying code outside Docker, but running inside
==============================================================
docker run --rm -it --net=host -v $PWD/examples:/home/examples -v $PWD/code/openmalaria/:/code/openmalaria bradleygh/openmalariapp


When building the prob prog version
===================================

When openMalaria is being built it is actively looking for the current version of schema, in this case the schema
version is 39.0, If the main directory name is not called "openmalaria_schema_<version_number> then the code will fail to build.
In addition to this, as specified by the openMalaria readme, you will have to change
all  the relevant places in the script where schema number appears before a build.
Seems very inefficient, but that is the way in whcih the simulator is set up.


Running OM simulator with Pyprob
================================

Mac
===
docker run --rm -it -p 2345:2345 -v $PWD/examples:/home/examples -v $PWD/code/openmalaria/:/code/openmalaria bradleygh/openmalariapp

add when calling ./openmalaria
$ ./openMalaria tcp://*:2345

Linux
=====
docker run --rm -it -v $PWD/examples:/home/examples -v $PWD/code/openmalaria/:/code/openmalaria bradleygh/openmalariapp

add when calling ./openmalaria
$ ./openMalaria ipc://@<some_string>


Using Singluarity instead
=========================

To convert a dockerfile to singularityfile run:

pip install singularity

Then in the terminal / commmand line run:

spython recipe Dockerfile >> Singularity

This will convert the Dockerfile to a singularity file and save the output as Singularity.

We can also make use of pre-built docker containers, without having to install docker, by running
the following:

singularity pull docker://bradleygh:openmalariapp