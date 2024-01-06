"""Constants used in the Python version of OpenPTV."""
NMAX = 10000  # correspondences
MAXCAND = 40  # assuming a maximum capacity of MAXCAND candidates
PT_UNUSED = -999

COORD_UNUSED = -1e10

# These define the structure of the sigma array returned from orient()
IDT = 10
NPAR = 19
CONVERGENCE = 0.00001

# Tracking
POSI = 80
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2
CORRES_NONE = -1

SORTGRID_EPS = 25

TR_UNUSED = -1
TR_BUFSPACE = 4  # 4 frames in the buffer to track
TR_MAX_CAMS = 4  # 4 cameras in the imaging system
MAX_TARGETS = 20480  # maximum number of targets
MAX_CANDS = 4
ADD_PART = 3


NUM_ITER = 80
POS_INF = 1e20
CONVERGENCE = 0.0001
