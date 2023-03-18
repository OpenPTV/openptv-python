import os


class mm_np:
    def __init__(self):
        self.nlay = 0
        self.n1 = 0.0
        self.n2 = [0.0, 0.0, 0.0]
        self.d = [0.0, 0.0, 0.0]
        self.n3 = 0.0


# Define sequence_par as a struct with properties num_cams, img_base_name, first and last. Define functions read_sequence_par, new_sequence_par, free_sequence_par, and compare_sequence_par.
class sequence_par:
    def __init__(self, num_cams=1, img_base_name="img/cam1.", first=0, last=1):
        self.num_cams = num_cams
        self.img_base_name = img_base_name
        self.first = first
        self.last = last


# Define track_par as a struct with properties dacc, dangle, dvxmax, dvxmin, dvymax, dvymin, dvzmax, dvzmin, dsumg, dn, dnx, dny, add. Define functions read_track_par and compare_track_par.
class track_par:
    def __init__(
        self,
        dacc=0.0,
        dangle=100.0,
        dvxmax=1.0,
        dvxmin=0.0,
        dvymax=1.0,
        dvymin=0.0,
        dvzmax=1.0,
        dvzmin=0.0,
        dsumg=1.0,
        dn=1.0,
        dnx=1.0,
        dny=1.0,
        add=0,
    ):
        self.dacc = dacc
        self.dangle = dangle
        self.dvxmax = dvxmax
        self.dvxmin = dvxmin
        self.dvymax = dvymax
        self.dvymin = dvymin
        self.dvzmax = dvzmax
        self.dvzmin = dvzmin
        self.dsumg = dsumg
        self.dn = dn
        self.dnx = dnx
        self.dny = dny
        self.add = add


# Define volume_par as a struct with properties X_lay, Zmin_lay, Zmax_lay, cn, cnx, cny, csumg, eps0, corrmin. Define functions read_volume_par and compare_volume_par.
class volume_par:
    def __init__(
        self,
        X_lay=(-100.0, 100.0),
        Zmin_lay=-10,
        Zmax_lay=10,
        cn=1,
        cnx=1,
        cny=1,
        csumg=1,
        eps0=0.01,
        corrmin=0.2,
    ):
        self.X_lay = X_lay
        self.Zmin_lay = Zmin_lay
        self.Zmax_lay = Zmax_lay
        self.cn = cn
        self.cnx = cnx
        self.cny = cny
        self.csumg = csumg
        self.eps0 = eps0
        self.corrmin = corrmin


# Define control_par as a struct with properties num_cams, img_base_name, cal_img_base_name, hp_flag, allCam_flag, tiff_flag, imx, imy, pix_x, pix_y, chfield and mm. Define functions new_control_par, read_control_par, and free_control_par.
class control_par:
    def __init__(
        self,
        num_cams=1,
        img_base_name="img/cam1.",
        cal_img_base_name="cal/cam1.",
        hp_flag=1,
        allCam_flag=0,
        tiff_flag=1,
        imx=1024,
        imy=1024,
        pix_x=0.01,
        pix_y=0.01,
        chfield=1,
        mm=mm_np(),
    ):
        self.num_cams = num_cams
        self.img_base_name = img_base_name
        self.cal_img_base_name = cal_img_base_name
        self.hp_flag = hp_flag
        self.allCam_flag = allCam_flag
        self.tiff_flag = tiff_flag
        self.imx = imx
        self.imy = imy
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.chfield = chfield
        self.mm = mm

    def set_image_size(self, imx, imy):
        self.imx = imx
        self.imy = imy

    def set_pixel_size(self, pix_x, pix_y):
        self.pix_x = pix_x
        self.pix_y = pix_y


# Define target_par as a struct with properties discont, gvthres, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min, cr_sz.
class target_par:
    def __init__(
        self,
        discont=100,
        gvthres=100,
        nnmin=1,
        nnmax=100,
        nxmin=1,
        nxmax=100,
        nymin=1,
        nymax=100,
        sumg_min=10,
        cr_sz=1,
    ):
        self.discont = discont
        self.gvthres = gvthres
        self.nnmin = nnmin
        self.nnmax = nnmax
        self.nxmin = nxmin
        self.nxmax = nxmax
        self.nymin = nymin
        self.nymax = nymax
        self.sumg_min = sumg_min
        self.cr_sz = cr_sz


def read_sequence_par(filename, num_cams):
    SEQ_FNAME_MAX_LEN = 200
    line = [""] * SEQ_FNAME_MAX_LEN

    try:
        par_file = open(filename, "r")

    except:
        return None

    ret = new_sequence_par(num_cams)

    for cam in range(num_cams):
        line = par_file.readline().strip()

        ret.img_base_name[cam] = line

    ret.first = int(par_file.readline().strip())
    ret.last = int(par_file.readline().strip())

    par_file.close()
    return ret


def new_sequence_par(num_cams):
    # Allocate memory for the new sequence_par struct
    ret = sequence_par()
    ret.img_base_name = [None] * num_cams
    ret.num_cams = num_cams

    # Allocate memory for the inner pointers of sequence_par struct
    for cam in range(num_cams):
        ret.img_base_name[cam] = str()

    return ret


def compare_sequence_par(sp1, sp2):
    cam = 0

    if sp1.first != sp2.first or sp1.last != sp2.last or sp1.num_cams != sp2.num_cams:
        return 0  # not equal

    while cam < sp1.num_cams:
        if sp1.img_base_name[cam] != sp2.img_base_name[cam]:
            return 0  # not equal

        cam += 1

    return 1  # equal


def read_track_par(filename: str) -> track_par:
    """Read tracking parameters from file and return track_par object."""
    track = track_par()
    try:
        with open(filename, "r", encoding="utf-8") as fpp:
            track.dvxmin = float(fpp.readline().rstrip())
            track.dvxmax = float(fpp.readline().rstrip())
            track.dvymin = float(fpp.readline().rstrip())
            track.dvymax = float(fpp.readline().rstrip())
            track.dvzmin = float(fpp.readline().rstrip())
            track.dvzmax = float(fpp.readline().rstrip())
            track.dangle = float(fpp.readline().rstrip())
            track.dacc = float(fpp.readline().rstrip())
            track.add = int(fpp.readline().rstrip())
            track.dsumg = track.dn = track.dnx = track.dny = 0
            return track
    except IOError:
        print(f"Error reading tracking parameters from {filename}")
        return None


def compare_track_par(t1, t2) -> bool:
    """Compare two track_par objects.

    Args:
    ----
        t1 (track_par): track_par object
        t2 (track_par): track_par object

    Returns:
    -------
        bool: True of False
    """
    return (
        (t1.dvxmin == t2.dvxmin)
        and (t1.dvxmax == t2.dvxmax)
        and (t1.dvymin == t2.dvymin)
        and (t1.dvymax == t2.dvymax)
        and (t1.dvzmin == t2.dvzmin)
        and (t1.dvzmax == t2.dvzmax)
        and (t1.dacc == t2.dacc)
        and (t1.dangle == t2.dangle)
        and (t1.dsumg == t2.dsumg)
        and (t1.dn == t2.dn)
        and (t1.dnx == t2.dnx)
        and (t1.dny == t2.dny)
        and (t1.add == t2.add)
    )


def read_volume_par(filename):
    """Reads volume parameters from file and returns volume_par object.

    Args:
    ----
        filename (str): filename

    Returns:
    -------
        volume_par: volume of interest parameters
    """
    ret = volume_par()

    with open(filename, "r", encoding="utf-8") as fpp:
        lines = fpp.readlines()

    ret.X_lay[0] = float(lines[0])
    ret.Zmin_lay[0] = float(lines[1])
    ret.Zmax_lay[0] = float(lines[2])
    ret.X_lay[1] = float(lines[3])
    ret.Zmin_lay[1] = float(lines[4])
    ret.Zmax_lay[1] = float(lines[5])
    ret.cnx = float(lines[6])
    ret.cny = float(lines[7])
    ret.cn = float(lines[8])
    ret.csumg = float(lines[9])
    ret.corrmin = float(lines[10])
    ret.eps0 = float(lines[11])

    return ret


def compare_volume_par(v1, v2):
    return (
        (v1.X_lay[0] == v2.X_lay[0])
        and (v1.Zmin_lay[0] == v2.Zmin_lay[0])
        and (v1.Zmax_lay[0] == v2.Zmax_lay[0])
        and (v1.X_lay[1] == v2.X_lay[1])
        and (v1.Zmin_lay[1] == v2.Zmin_lay[1])
        and (v1.Zmax_lay[1] == v2.Zmax_lay[1])
        and (v1.cn == v2.cn)
        and (v1.cnx == v2.cnx)
        and (v1.cny == v2.cny)
        and (v1.csumg == v2.csumg)
        and (v1.corrmin == v2.corrmin)
        and (v1.eps0 == v2.eps0)
    )


def new_control_par(cams):
    """Creates a set of new control parameters for a given number of cameras.

    Args:
    ----
        cams (int): number of cameras

    Returns:
    -------
        control_par: control parameters object
    """
    return control_par(num_cams=cams)


def read_control_par(filename):
    if not os.path.isfile(filename):
        print("Could not open file " + filename)
        return None

    with open(filename, "r") as par_file:
        num_cams = int(par_file.readline().strip())
        ret = control_par(num_cams=num_cams)

        for cam in range(ret.num_cams):
            ret.img_base_name[cam] = par_file.readline().strip()

            ret.cal_img_base_name[cam] = par_file.readline().strip()

        ret.hp_flag = int(par_file.readline().strip())
        ret.allCam_flag = int(par_file.readline().strip())
        ret.tiff_flag = int(par_file.readline().strip())
        ret.imx = int(par_file.readline().strip())
        ret.imy = int(par_file.readline().strip())
        ret.pix_x = float(par_file.readline().strip())
        ret.pix_y = float(par_file.readline().strip())
        ret.chfield = int(par_file.readline().strip())
        ret.mm.n1 = float(par_file.readline().strip())
        ret.mm.n2[0] = float(par_file.readline().strip())
        ret.mm.n3 = float(par_file.readline().strip())
        ret.mm.d[0] = float(par_file.readline().strip())

        ret.mm.nlay = 1

    return ret


def free_control_par(cp):
    cam = 0
    while cam < cp.num_cams:
        del cp.img_base_name[cam]
        cp.img_base_name[cam] = None
        del cp.cal_img_base_name[cam]
        cp.cal_img_base_name[cam] = None
        cam += 1

    del cp.img_base_name
    cp.img_base_name = None

    del cp.cal_img_base_name
    cp.cal_img_base_name = None

    del cp.mm
    cp.mm = None

    del cp


def compare_control_par(c1, c2):
    cam = 0
    if c1.num_cams != c2.num_cams:
        return 0

    while cam < c1.num_cams:
        if c1.img_base_name[cam] != c2.img_base_name[cam]:
            return 0
        if c1.cal_img_base_name[cam] != c2.cal_img_base_name[cam]:
            return 0
        cam = cam + 1

    if (
        c1.hp_flag != c2.hp_flag
        or c1.allCam_flag != c2.allCam_flag
        or c1.tiff_flag != c2.tiff_flag
        or c1.imx != c2.imx
        or c1.imy != c2.imy
        or c1.pix_x != c2.pix_x
        or c1.pix_y != c2.pix_y
        or c1.chfield != c2.chfield
        or compare_mm_np(c1.mm, c2.mm) == 0
    ):
        return 0

    return 1


def compare_mm_np(mm_np1, mm_np2):
    if mm_np1.n2[0] != mm_np2.n2[0] or mm_np1.d[0] != mm_np2.d[0]:
        return 0
    if mm_np1.nlay != mm_np2.nlay or mm_np1.n1 != mm_np2.n1 or mm_np1.n3 != mm_np2.n3:
        return 0
    return 1


def read_target_par(filename):
    ret = None
    try:
        with open(filename, "r") as file:
            ret = target_par()
            ret.gvthres = [0, 0, 0, 0]
            ret.nnmin = 0
            ret.nnmax = 0
            ret.nxmin = 0
            ret.nxmax = 0
            ret.nymin = 0
            ret.nymax = 0

            line = file.readline().split()
            if line:
                ret.gvthres[0] = int(line[0])

            line = file.readline().split()
            if line:
                ret.gvthres[1] = int(line[0])

            line = file.readline().split()
            if line:
                ret.gvthres[2] = int(line[0])

            line = file.readline().split()
            if line:
                ret.gvthres[3] = int(line[0])

            line = file.readline().split()
            if line:
                ret.discont = int(line[0])

            line = file.readline().split()
            if len(line) == 2:
                ret.nnmin = int(line[0])
                ret.nnmax = int(line[1])

            line = file.readline().split()
            if len(line) == 2:
                ret.nxmin = int(line[0])
                ret.nxmax = int(line[1])

            line = file.readline().split()
            if len(line) == 2:
                ret.nymin = int(line[0])
                ret.nymax = int(line[1])

            line = file.readline().split()
            if line:
                ret.sumg_min = int(line[0])

            line = file.readline().split()
            if line:
                ret.cr_sz = int(line[0])

            file.close()
            return ret

    except IOError:
        print("Could not open target recognition parameters file %s.\n", filename)
        return None


def compare_target_par(targ1, targ2):
    return (
        targ1.discont == targ2.discont
        and targ1.gvthres[0] == targ2.gvthres[0]
        and targ1.gvthres[1] == targ2.gvthres[1]
        and targ1.gvthres[2] == targ2.gvthres[2]
        and targ1.gvthres[3] == targ2.gvthres[3]
        and targ1.nnmin == targ2.nnmin
        and targ1.nnmax == targ2.nnmax
        and targ1.nxmin == targ2.nxmin
        and targ1.nxmax == targ2.nxmax
        and targ1.nymin == targ2.nymin
        and targ1.nymax == targ2.nymax
        and targ1.sumg_min == targ2.sumg_min
        and targ1.cr_sz == targ2.cr_sz
    )


def write_target_par(targ, filename):
    file = open(filename, "w")

    if file is None:
        print(f"Can't create file: {filename}")

    file.write(
        f"{targ.gvthres[0]}\n{targ.gvthres[1]}\n{targ.gvthres[2]}\n{targ.gvthres[3]}\n{targ.discont}\n{targ.nnmin}\n{targ.nnmax}\n{targ.nxmin}\n{targ.nxmax}\n{targ.nymin}\n{targ.nymax}\n{targ.sumg_min}\n{targ.cr_sz}"
    )

    file.close()
