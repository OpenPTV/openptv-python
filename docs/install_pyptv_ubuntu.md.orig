## How to install OpenPTV-Python version on Ubuntu 12.04

NOTE that steps 1-2 can be replaced by installing the *academic license* version of CANOPY from Enthought, see <https://www.enthought.com/products/canopy/>. It is also possible to get Canopy Express and add packages manually.


1. Following this link I got all the required packages for Ubuntu whih are not installed by default, with few updates. 

original link: <neuroimaging.scipy.org/doc/manual/html/devel/tools/virtualenv-tutor.html>

On Ubuntu we use package manager (`sudo apt-get install package-name`): 


		sudo apt-get install python-numpy python-scipy python-setuptools python-wxgtk2.8 cython swig  
		python-imaging python-vtk python-pyrex python-matplotlib g++ libc6-dev libx11-dev python2.7-dev 
		freeglut3-dev libtool git python-sphinx curl make cmake python-enable

if you get some error about `Gl/gl.h` or `GL/glu.h`, maybe also:

`sudo apt-get install libatlas-base-dev libgl1--mesa-dev libglu1-mesa-dev`


2. Get the necessary packages from Github:  

ETS from Enthought:

* Download the Python script that installs all-in-one `ets.py`:  

<https://github.com/enthought/ets/raw/master/ets.py>  

  	$ mkdir ets
		$ cd ets
		$ python ets.py clone 
		$ sudo python ets.py install

3. Download and install Check unit testing framework:  

Download from <http://sourceforge.net/projects/check/files/latest/download?source=files> or use:

		$ curl -O http://sourceforge.net/projects/check/files/latest/download?source=files
		$ ./configure 
		$ make
		$ make check
		$ sudo make install 

4. Download and install `liboptv`:  

		$ git clone git://github.com/OpenPTV/openptv.git
		$ cd openptv/liboptv
		$ cmake -G "Unix Makefiles" -i
		CMake will ask you some questions about your system (accepting the 
		defaults is usually ok)
		$ make
		$ make check
		$ sudo make install
		$ cd ~

5. Download and install `openptv-python`:  

		$ git clone git://github.com/OpenPTV/openptv-python.git
		$ cd openptv-python/pyptv_gui
		$ python setup.py build_ext --inplace

6. For the test, get the test_folder and see if it works:
		$ cd ~
		$ git clone https://bitbucket.org/turbulencelabtau/ptv_test_folder.git
		$ cd ptv_test_folder
		$ mkdir res
		$ cd ~/openptv-python/pyptv-gui/ 
		$ python pyptv_gui.py ~/ptv_test_folder/test/


For Mac OS X I found this Gist very useful: <https://gist.github.com/satra/845545> and another copy here: <https://www.ibic.washington.edu/wiki/display/faq/How+do+I+install+EPD+Python+in+a+virtualenv>














