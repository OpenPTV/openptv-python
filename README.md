"old" Python GUI for OpenPTV
============================

**OpenPTV-Python** is the Python graphical user interface (GUI) for the [OpenPTV](http://www.openptv.net) library `liboptv`. It is basically the Traits, TraitsUI, Chaco, Enable based GUI (most are Enthought Inc. packages) that *interfaces* the OpenPTV library that includes all the core algorithms (correspondence, tracking, calibration, etc.) written in ANSI C. 

Both the GUI and the OpenPTV library are in the development phase and continuously refactored. Please follow the development on the community mailing list:

	openptv@googlegroups.com

## Note that this GUI version is outdated, we move to PyPTV
https://github.com/alexlib/pyptv

## Documentation, including installation instructions

<http://openptv-python.readthedocs.io>

## Getting started:

If the compilation passed, open the terminal and run:  

		python pyptv_gui/pyptv_gui.py test_cavity
		
or:  

		pythonw pyptv_gui/pyptv_gui.py test_cavity
		
It is possible to install wxPython instead of PyQt4, and switch between those:  

		ETS_TOOLKIT=wx python pyptv_gui.py test_cavity

Follow the instructions in our **screencasts and tutorials**:
  
  *  Tutorial 1: <http://youtu.be/S2fY5WFsFwo>  
  
  *  Tutorial 2: <http://www.youtube.com/watch?v=_JxFxwVDSt0>   
  
  *  Tutorial 3: <http://www.youtube.com/watch?v=z1eqFL5JIJc>  
  
  
Ask for help on our mailing list:

	openptv@googlegroups.com

