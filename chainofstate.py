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

def readGauEner(text, natoms):
    """
    Read Gaussian output and find energy gradient.
    """
    ener = [i for i in text if "SCF Done:" in i]
    if len(ener) != 0:
        ener = ener[-1]
        ener = np.float64(ener.split()[4])
    else:
        ener = np.float64([i for i in text if "Energy=" in i][-1].split()[1])
    return ener * EH


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



def distance(a, b):
    """
    Calc the distance between two vectors.
    """
    return (((a - b) ** 2).sum()) ** 0.5


class InterpolatePath:

    def __init__(self, coords):
        self.dofFuncList = []
        dists = []
        for i in range(len(coords)):
            if i > 0:
                dists.append(distance(coords[i], coords[i-1]))
            else:
                dists.append(0.0)
        alphas = [sum(dists[:i+1])/sum(dists) for i in range(len(dists))]
        print(alphas)
        list1d = [i.ravel() for i in coords]
        for p in range(len(list1d[0])):
            k = [i[p] for i in list1d]
            self.dofFuncList.append(interpolate.interp1d(alphas, k, kind="cubic"))


    def getCoord(self, alpha):
        pos = np.zeros((len(self.dofFuncList,)))
        for n, f in enumerate(self.dofFuncList):
            pos[n] = f(alpha)
        return pos.reshape((-1,3))



def calcGauGrad(atom, crd, template, path="g09"):
    """
    Calculate gradient using Gaussian.
    """
    os.system("rm tmp.gjf tmp.log")
    with open("tmp.gjf", "w") as f:
        f.write(genQMInput(atom, crd, template))
    os.system("{} tmp.gjf".format(path))
    ener, grad = readFile("tmp.log", lambda x: readGauGrad(x, len(atom)))
    return ener, grad


def calcGauEner(atom, crd, template, path="g09"):
    """
    Calculate gradient using Gaussian.
    """
    os.system("rm tmp.gjf tmp.log")
    with open("tmp.gjf", "w") as f:
        f.write(genQMInput(atom, crd, template))
    os.system("{} tmp.gjf".format(path))
    ener = readFile("tmp.log", lambda x: readGauEner(x, len(atom)))
    return ener

# Chain-of-state




def runChainOfState(atom, xyzs, method="Euler", dT=0.01, templateEner="energy.gjf", templateForce="force.gjf"):
    # calc the energy and force of the first and last structure
    if method == "Euler":
        print("Euler step...")
        energyList = []
        gradList = []
        newPosList = []
        for nrep in range(len(xyzs)):
            print("%i"%nrep, end=" ")
            ener, grad = calcGauGrad(atom, xyzs[nrep], templateForce)
            energyList.append(ener)
            newPosList.append(xyzs[nrep] - dT * grad)
            print("%12.6f"%(ener/EH))
        print()
        
    elif method == "Runge-Kutta":
        energyList = []
        gradList = []
        newPosList = []
        for nrep in range(len(xyzs)):
            print(nrep, end="/")
            # calc k1
            ener, grad = calcGauGrad(atom, xyzs[nrep], templateForce)
            energyList.append(ener)
            k1 = dT * grad
            print("k1", end="/")
            # calc k2
            ener, grad = calcGauGrad(atom, xyzs[nrep] + 0.5 * k1, templateForce)
            k2 = dT * grad
            print("k2", end="/")
            # calc k3 
            ener, grad = calcGauGrad(atom, xyzs[nrep] + 0.5 * k2, templateForce)
            k3 = dT * grad
            print("k3", end="/")
            # calc k4
            ener, grad = calcGauGrad(atom, xyzs[nrep] + k3, templateForce)
            k4 = dT * grad
            print("k4")
            newPosList.append(xyzs[nrep] - 1.0/6.0 * k1 - 1.0/3.0 * k2 - 1.0/3.0 * k3 - 1.0/6.0 * k4)
    
    # build path
    path = InterpolatePath(newPosList)
    alpha = np.linspace(0.0, 1.0, len(newPosList))
    coordList = [path.getCoord(i) for i in alpha]

    return energyList, coordList


def argparse():
    CONF = None
    with open(sys.argv[1], "r") as f:
        try:
            string = "".join([i for i in f])
            CONF = json.loads(string)
        except:
            print("""
        Usage:
            python stringmethod.py config.json
                """)
            exit()
    if "dt" not in CONF:
        CONF["dt"] = 0.0001
    if "outEnergy" not in CONF:
        CONF["outEnergy"] = "energy.txt"
    if "method" not in CONF:
        CONF["method"] = "Euler"
    if "energyTempelate" not in CONF:
        CONF["energyTempelate"] = "energy.gjf"
    if "forceTempelate" not in CONF:
        CONF["forceTempelate"] = "force.gjf"
    if "initPath" not in CONF:
        print("""
    Error: Initial pathway need to be offered in XYZ format.
        """)
        exit()
    return CONF


def main():
    conf = argparse()
    
    xyz_list = readFile(conf["initPath"], readMultiXYZ)
    atom = xyz_list[0][0]
    coord = [i[1] * ANGSTROM for i in xyz_list]
    print(len(coord))
    # run cycles
    ncycle = 1
    while True:
        energy, coord = runChainOfState(atom, coord, method=conf["method"], templateEner=conf["energyTempelate"], templateForce=conf["forceTempelate"], dT=conf["dt"])
        with open(conf["outEnergy"], "a") as f:
            text = " ".join(["%16.8f"%(i/EH) for i in energy])
            text = text + "\n"
            f.write(text)
        crdname = "string-%i.xyz"%ncycle
        for nrep in range(len(energy)):
            writeXYZ(crdname, atom, coord[nrep], append=True, title="%16.8f"%(energy[nrep]/EH))
        ncycle += 1        


if __name__ == '__main__':
    main()
