import numpy as np
from scipy import interpolate
import os
import sys
import json

BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FS = 1e-15  # fs -> s
EH = 4.35974417e-18  # Hartrees -> J
KJ = 1000.0
KCAL = KJ * 4.184
H = 6.626069934e-34
KB = 1.38064852e-23
MOL = 6.02e23


def readFile(fname, func):
    """
    Read text file and deal with different functions.
    """
    with open(fname, "r") as f:
        text = [i for i in f if i.strip()]
    return func(text)


def readXYZ(text):
    """
    Read xyz format text.
    """
    natoms = int(text[0].strip())
    body = text[2:natoms + 2]
    body = [i.strip().split() for i in body]
    atom = [i[0] for i in body]
    crd = [[float(j) for j in i[1:]] for i in body]
    return atom, np.array(crd)


def readMultiXYZ(text):
    """
    Read XYZ file with multi conformations.
    """
    xyzs = []
    ip = 0
    while True:
        natom = int(text[ip].strip())
        xyzs.append(text[ip:ip + natom + 2])
        ip = ip + natom + 2
        if ip >= len(text):
            break
    return [readXYZ(i) for i in xyzs]


def readGauGrad(text, natoms):
    """
    Read Gaussian output and find energy gradient.
    """
    ener = [i for i in text if "SCF Done:" in i]
    if len(ener) != 0:
        ener = ener[-1]
        ener = np.float64(ener.split()[4])
    else:
        ener = np.float64([i for i in text if "Energy=" in i][-1].split()[1])
    for ni, li in enumerate(text):
        if "Forces (Hartrees/Bohr)" in li:
            break
    forces = text[ni + 3:ni + 3 + natoms]
    forces = [i.strip().split()[-3:] for i in forces]
    forces = [[np.float64(i[0]), np.float64(i[1]), np.float64(i[2])]
              for i in forces]
    return ener * EH, - np.array(forces) * EH / BOHR


def writeXYZ(fname, atom, xyz, title="Title", append=False, unit=ANGSTROM):
    """
    Write file with XYZ format. 
    """
    if append:
        f = open(fname, "a")
    else:
        f = open(fname, "w")
    f.write("%i\n" % len(atom))
    f.write("%s\n" % (title.rstrip()))
    for i in range(len(atom)):
        x, y, z = xyz[i, :]
        f.write("%s  %12.8f %12.8f %12.8f\n" %
                (atom[i], x / unit, y / unit, z / unit))
    f.close()


def genQMInput(atom, crd, temp):
    """
    Generate QM Input file for force calculation.
    """
    with open(temp, "r") as f:
        temp = f.readlines()
    wrt = []
    for line in temp:
        if "[title]" in line:
            wrt.append("Temparary input file\n")
        elif "[coord]" in line:
            for ni in range(len(atom)):
                wrt.append("%s  %16.8f %16.8f %16.8f\n" %
                           (atom[ni], crd[ni][0] / ANGSTROM, crd[ni][1] / ANGSTROM, crd[ni][2] / ANGSTROM))
        else:
            wrt.append(line)
    return "".join(wrt)


def genMassMat(atom):
    """
    Generate matrix of mass.
    """
    massd = {"H": 1.008,
             "C": 12.011,
             "N": 14.007,
             "O": 15.999,
             "S": 32.066,
             "CL": 35.453,
             "BR": 79.904, }
    massv = np.array([massd[i.upper()] for i in atom])
    massm = np.zeros((len(atom), 3))
    massm[:, 0] = massv
    massm[:, 1] = massv
    massm[:, 2] = massv
    return massm.reshape((-1, 3)) * AMU


def calcGauGrad(atom, crd, template, path="g09"):
    """
    Calculate gradient using Gaussian.
    """
    with open("tmp.gjf", "w") as f:
        f.write(genQMInput(atom, crd, template))
    os.system("{} tmp.gjf".format(path))
    grad = readFile("tmp.log", lambda x: readGauGrad(x, len(atom)))
    return grad

# Chain-of-state


def distance(a, b):
    """
    Calc the distance between two vectors.
    """
    return (((a - b) ** 2).sum()) ** 0.5


def decomp(a, b):
    # return va,vb
    # va + vb = a
    # va .* vb = 0
    # vb parrallel to b
    a, b = a.ravel(), b.ravel()
    m = np.dot(a, b) / np.dot(b, b)
    return (a - m * b).reshape((-1, 3)), (m * b).reshape((-1, 3))


def spring(a, b, k):
    """
    Generate spring force from a to b.
    """
    return (a - b) * 2. * k


def genPosFunc(pos_list, val_list):
    """
    Generate reaction pathway by interpolating and return pathway function.
    """
    list1d = [i.ravel() for i in pos_list]
    funclist = []
    for p in range(len(list1d[0])):
        k = [i[p] for i in list1d]
        funclist.append(interpolate.interp1d(val_list, k, kind="cubic"))

    def posfunc(alpha):
        pos = np.zeros((len(funclist,)))
        for n, f in enumerate(funclist):
            pos[n] = f(alpha)
        return pos.reshape((-1, 3))
    return posfunc


def numGrad(posfunc, alpha):
    """
    Numerically calc the gradient at each point.
    """
    delta = 0.0001
    grad1d = (posfunc(alpha + delta) - posfunc(alpha - delta)) / 2. / delta
    return grad1d


def genLinearConf(start, end, points):
    """
    Interpolate conformations between start and end confs. 
    points :: The total conformations on the linear pathway. 
              Return [start, end] is points is smaller than 2.
    """
    if points < 2:
        return [start, end]
    confs = []
    for i in range(points):
        confs.append(start * (1 - i / (points - 1)) + end * i / (points - 1))
    return confs


def runChainOfState(atom, xyzs, method="NEB", inite=None, jobname="", template=None, Kspring=10.0 * KCAL / MOL / ANGSTROM**2, LRate=0.5, Rmax=0.1 * ANGSTROM, dR=0.1 * ANGSTROM, maxcycle=100):
    # calc the energy of the first and last structure
    energy = np.zeros((len(xyzs),))
    if inite is not None:
        efirst = inite[0]
        elast = inite[-1]
    else:
        efirst, _ = calcGauGrad(atom, xyzs[0], template)
        elast, _ = calcGauGrad(atom, xyzs[-1], template)
    coord = [np.zeros(i.shape) for i in xyzs]
    new_coord = [np.zeros(i.shape) for i in xyzs]
    for n, i in enumerate(xyzs):
        coord[n][:] = i[:]
        new_coord[n][:] = i[:]
    pe_grad = [np.zeros(i.shape) for i in xyzs]
    path_grad = [np.zeros(i.shape) for i in xyzs]
    massm = genMassMat(atom)
    energy[0] = efirst
    energy[-1] = elast
    moves = np.zeros(energy.shape)
    for ncycle in range(1, maxcycle + 1):
        print("Cycle %i" % ncycle)
        # calc gradient of PE
        for nimage in range(len(xyzs)):
            if nimage != 0 and nimage != len(xyzs) - 1:
                e, g = calcGauGrad(atom, coord[nimage], template)
                energy[nimage] = e
                pe_grad[nimage] = g
            print("E: %.4f  " %
                  ((energy[nimage] - energy[0]) / (KCAL / MOL)), end="  ")
        print()
        # get pathway
        alpha = [0.0]
        for n in range(1, len(xyzs)):
            alpha.append(alpha[n - 1] + distance(coord[n - 1], coord[n]))
        alpha = [i / alpha[-1] for i in alpha]
        posfunc = genPosFunc(coord, alpha)
        if "NEB" in method:
            grad_sum = [np.zeros(i.shape) for i in xyzs]

            # get direction of NEB gradient
            for n, a in enumerate(alpha):
                if n == 0 or n == len(alpha) - 1:
                    continue
                path_grad[n] = numGrad(posfunc, a)
            # calc elastic band grad
            neb_grad = [np.zeros(i.shape) for i in xyzs]
            for i in range(1, len(xyzs) - 1):
                k1 = spring(coord[i], coord[i - 1], Kspring)
                k2 = spring(coord[i], coord[i + 1], Kspring)
                neb_grad[i] = k1 + k2
            # summarize PE grad and NEB grad
            for i in range(1, len(xyzs) - 1):
                gv, gh = decomp(pe_grad[i], path_grad[i])
                if "CI" in method:
                    if i == np.argmax(energy):
                        gh = - gh
                else:
                    _, gh = decomp(neb_grad[i], path_grad[i])
                gsum = gv + gh
                moves[i] = np.sqrt(np.power(Rstep * gsum, 2).sum())
                grad_sum[i] = gsum
                if max(moves) > Rmax:
                    scale = Rmax / max(moves)
                else:
                    scale = 1.0
            # renew position
            for i in range(1, len(xyzs) - 1):
                new_coord[i] = coord[i] - Rstep * grad_sum[i] * scale
        print(moves / ANGSTROM)
        # output pathway
        for i in range(len(xyzs)):
            title = "Cycle %i    Image %i    Energy:%.6f\n" % (
                ncycle, i, energy[i] / EH)
            writeXYZ("%s-pathway%i.xyz" % (jobname, ncycle), atom,
                     coord[i], title=title, append=True)
        # renew coords
        for i in range(len(coord)):
            coord[i] = new_coord[i]
    return energy, coord


def argparse():
    CONF = None
    for n, i in enumerate(sys.argv):
        if i == "-j" or i == "--json":
            with open(sys.argv[n + 1], "r") as f:
                CONF = json.loads("".join([i for i in f]))
    if CONF is None:
        print("""
    Usage:
        python sot.py -j/--json config.json
            """)
    for NC in range(len(CONF["workflow"])):
        if "Kspring" not in CONF["workflow"][NC]["parameter"]:
            CONF["workflow"][NC]["parameter"]["Kspring"] = 10.0
        if "LRate" not in CONF["workflow"][NC]["parameter"]:
            CONF["workflow"][NC]["parameter"]["LRate"] = 0.001
        if "Rmax" not in CONF["workflow"][NC]["parameter"]:
            CONF["workflow"][NC]["parameter"]["Rmax"] = 0.10
        if "maxcycle" not in CONF["workflow"][NC]["parameter"]:
            CONF["workflow"][NC]["parameter"]["maxcycle"] = 30
    jobnames = [i["jobname"] for i in CONF["workflow"]]
    if len(jobnames) != len(list(set(jobnames))):
        print("""
    The name of each job should be different or the output file will be messy.
            """)
    return CONF


def main():
    conf = argparse()
    # generate init confs
    # if there are 2 or 3 images, then doing linear interpole
    # if there are 4 images or more, do cubic
    xyz_list = readFile(conf["coord"], readMultiXYZ)
    atom = xyz_list[0][0]
    xyzs = [i[1] * ANGSTROM for i in xyz_list]
    totalI = conf["index"][-1] - conf["index"][0]
    if len(xyzs) == 2:
        coord = genLinearConf(xyzs[0], xyzs[1], totalI)
    elif len(xyzs) == 3:
        part1 = genLinearConf(xyzs[0], xyzs[1], conf["index"][1] - conf["index"][0] + 1)
        part2 = genLinearConf(xyzs[1], xyzs[2], conf["index"][2] - conf["index"][1] + 1)
        coord = part1[:-1] + part2[1:]
    else:
        # get pathway
        alpha = [0.0]
        for n in range(1, len(xyzs)):
            alpha.append(alpha[n - 1] + distance(xyzs[n - 1], xyzs[n]))
        alpha = [i / alpha[-1] for i in alpha]
        posfunc = genPosFunc(xyzs, alpha)
        new_alpha = [0.0]
        for n, a in enumerate(alpha):
            if n == 0:
                continue
            i_end = conf["index"][n]
            i_start = conf["index"][n - 1]
            a_start = alpha[n - 1]
            a_end = a
            for p in range(i_end - i_start):
                new_alpha.append((a_end - a_start) /
                                 (i_end - i_start) * p + a_start)
        new_alpha.append(1.0)
        coord = [posfunc(a) for a in new_alpha]
        coord[0] = xyzs[0]
        coord[-1] = xyzs[-1]
    # run cycles
    for n, param in enumerate(conf["workflow"]):
        if n == 0:
            energy, coord = runChainOfState(atom, coord, method=param["method"], template=conf["template"], Kspring=param[
                                            "Kspring"] * KCAL / MOL / ANGSTROM**2, LRate=param["LRate"], Rmax=param["Rmax"] * ANGSTROM, maxcycle=param["maxcycle"])
        else:
            energy, coord = runChainOfState(atom, coord, inite=energy, method=param["method"], template=conf["template"], Kspring=param[
                                            "Kspring"] * KCAL / MOL / ANGSTROM**2, LRate=param["LRate"], Rmax=param["Rmax"] * ANGSTROM, maxcycle=param["maxcycle"])


if __name__ == '__main__':
    main()
