
=================================================
Refactoring plan for improving PyPTV code quality
=================================================

In the file docs/todo.txt the interested reader may find several ideas on how
to improve the scientific value of PyPTV. There is certainly a lot of
potential, but it will not be easy to realize, due to existing code quality
problems, which also affect the operation of existing features. In this file 
I present the tasks needed in order to bring the code up to quality standards
in usability, maintainability, readability, trustworthiness and extensibility.

Each task below has an explanation and a status description. The status 
description should be updated whenever a task is changed.

Yosef Meller.


Factor out duplicated code
--------------------------
There are repeated code pieces for some things. Repetition is bad because if
you find a problem or want to add a feature, you must do it consistently across
all copies. It is inevitable to forget some copies sometime, leading to
divergent code having slightly different problems and features at each place it
is used. Furthermore, each reuse requires a new copy of the entire 
functionality.

The task is to identify duplication and move into common functions. Minor
differences should be settled by removal or parameterization.

Status:
~~~~~~~
By now I've identified duplication related to handling of the frame-buffer 
variables (t4, c4, mega): in allocation, filling from text files, and writing
results. Other duplications are yet to be identified. 

The frame-buffer variables are no longer needed by the forward-tracking code,
which uses the new proper framebuf class (in tracking_frame_buf.{c,h}). The 
other tracking routines still use them, but all related duplication in ptv.c
has been collapsed into functions in tracking_frame_buf.c.

In tracking.c there are many repeating or almost-repeating snippets that can be
made into functions. Already started with {reset,copy}_foundpix_array().

Removal of global variables
---------------------------
Almost all system state is global. Since the state is used extensively, it is
changed by several parts of the program without coordination. So far one
result identified is the use of assumed-initialized memory without
initializing it. Other possible misuses of globals are documented in extensive
literature online, and I fully expect to encounter them at some point.

Globals must be replaced by discrete objects passed in as parameters to
functions, each holding only the subset of system variables that the function
needs. Our strength is in our disunity. The caller (Python) should be tasked
with keeping track of the relevant state.

Status: (t4, c4, mega) are no longer used in forward tracking, but still used
in backward tracking and in the unused algorithm in ptv.c. There's a start of
handling the calibration globals in calibration.{c,h}.


Testing
-------
Currently there are very few unit tests. The task is to find a unit-testing 
framework that works for C (maybe nosetests can be used, who knows), and start
increasing coverage.

Status: Selected Check as the C testing framework. All new code is tested
except some recent functions that just removed duplication in track.c. The
processing workflow (sequencing, tracking, tracking back) is tested against
reference results using Python unit-tests in pyptv_gui/test/test_processing.py


Error handling
--------------
Errors are currently handled by printing a message and then doing nothing. The
result is crashes somewhere away from the actual error, where the error becomes
relevant. 

Functions which can fail should be modified to return their error state, and
that state must be checked by callers and handled appropriately.

Status: All frame-buffer functions return an error state if needed, but the 
callers do not check this right now, so the error printing is kept. however, we
now return on any sign of error, rather than try to write to a file we 
couldn't open.


Hygiene
-------
Meaningful variable names.
Indentation is mostly OK, but needs work.
Paragraphs. 
Janitoring.

Status: not started.

