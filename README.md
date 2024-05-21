# The role of Levodopa and Carbidopa in Parkinson's disease using Repast4Py: A simulation approach

This is the GitHub repository for the project of Francesco Finucci for the exam of MultiAgent Systems Lab by Prof. Emanuela Merelli and Prof. Stefano Maestri at the university of Camerino.

The goal of this project was to create a MAS simulator using Repast4Py that is able to replicate a certain aspect of Parkinson's disease in a multi-environment setting. My choice was to study the role of two medicines called **levodopa** and **carbidopa** and how the communication between the Substantia Nigra (the region of the brain in which Parkinson develops) and the peripheral immune system works in 3 different conditions:
- Without levodopa and without carbidopa;
- With levodopa but without carbidopa;
- With both levodopa and carbidopa.

## How to use
Python 3.8 or above is a requirement.

In order to setup this project on your machine, you first have to configure and install Repast4Py. This can be easily done by following this [Repast4Py installation guide](https://repast.github.io/repast4py.site/guide/user_guide.html#_getting_started).
Then, you can just install all of the python3 dependencies by running the command
```
pip install -r requirements.txt
```
or by installing the requirements using another python container such as _venv_ or _conda_.

## How to run the simulation and the GUI
In order to run the simulation you have to use the following command
```
mpirun -n 4 python src/parkinson.py config/parkinson_model.yaml
```
This will generate some CSV files in the output/ folder that will be then used by the GUI to animate the simulation. To run the GUI, run
```
python3 src/gui.py
```
Remember to clean the /output folder each and everytime you need to re-run the simulation, since the CSV files won't update automatically.
