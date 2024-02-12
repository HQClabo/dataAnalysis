# Installation with as an editable package
If you want to install this package with pip, you can clone the github repository to where ever you like and open your terminal (the anaconda if you are using anaconda) in the pyhton environment where you want to install it. Then navigate to the location of the package and run
	pip install -e .
Don't forget the dot at the end. This will run setup.py and install the package in an editable format. Effectively, it will create an egg folder (dataAnalysis.egg-info in this specific case), which will link the git repository to pip and make sure pip knows that the package is installed, i.e. it will show when calling `pip list`.

# Installation by running setup.py
Alternatively, you can follow the same steps (cloning the repository and navigating to it in the terminal) but then run setup.py directly using `python setup.py`. However, this will not make it appear on `pip list`. 
