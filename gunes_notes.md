In the dir with notes_for_gunes run the following:

$: docker run --rm -it --net=host -v $PWD/examples:/home/examples -v $PWD/code/:/code/ bradleygh/openmalariapp

Once in the docker run

$: tmux  # Split screen 
$: jupyter notebook --allow-root  # in one tmux screen and navigate to /code/notebooks/example.ipynb
$: cd /home/examples/test # in another tmux screen

Then proceed as normal. 

