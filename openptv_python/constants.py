"""Constants used in the Python version of OpenPTV."""
NMAX = 202400
MAXCAND = 200  # assuming a maximum capacity of MAXCAND candidates
PT_UNUSED = -999

COORD_UNUSED = -1e10

# These define the structure of the sigma array returned from orient()
IDT = 10
NPAR = 19


# Tracking
POSI = 80
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2
CORRES_NONE = -1


TR_UNUSED = -1
TR_BUFSPACE = 4
TR_MAX_CAMS = 4
MAX_TARGETS = 20000
MAX_CANDS = 4
ADD_PART = 3


NUM_ITER = 80
POS_INF = 1e20
CONVERGENCE = 0.00001
