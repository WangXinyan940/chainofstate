import numpy as np 
from scipy import interpolate
import os,sys
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
        f.write("%s  %12.8f %12.8f %12.8f\n" % (atom[i], x / unit, y / unit, z / unit))
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
    os.system("cp tmp.chk old.chk")
    return grad

#Chain-of-state
def distance(a,b): 
    """
    Calc the distance between two vectors.
    """
    return (((a - b) ** 2).sum()) ** 0.5 
 
def decomp(a,b):
    #return va,vb
    #va + vb = a
    #va .* vb = 0
    #vb parrallel to b
    a, b = a.ravel(), b.ravel()
    m = np.dot(a,b) / np.dot(b,b)
    return (a - m * b).reshape((-1,3)), (m * b).reshape((-1,3))


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
    for p in range(list1d.shape[0]):
        k = [i[p] for i in list1d]
        funclist.append(interpolate.interp1d(val_list,k,kind="cubic"))
    def posfunc(alpha):
        pos = np.zeros((len(funclist,)))
        for n,f in enumerate(funclist):
            pos[n] = f(alpha)
        return pos.reshape((-1, 3))
    return posfunc


def numGrad(posfunc, alpha):
    """
    Numerically calc the gradient at each point.
    """
    delta = 0.002
    grad1d = (posfunc(alpha + delta) - posfunc(alpha - delta)) / 2. / delta
    return grad1d.reshape((-1, 3))


def genLinearConf(start, end, points):
    confs = []
    for i in range(points + 1):
        confs.append(start * (1 - i/points) + end * i/points)
    return confs 


def runChainOfState(atom, xyzs, method="NEB", force=None, template=None, Kspring=10.0*KCAL/MOL/ANGSTROM**2, Rstep=0.5, dG=-1.*KCAL/MOL, dR=0.1*ANGSTROM, maxcycle):
    # calc the energy of the first and last structure
    efirst, _ = calcGauGrad(atom, xyzs[0], template)
    elast, _ = calcGauGrad(atom, xyzs[-1], template)
    energy = np.zeros((len(xyzs),))
    coord = [np.zeros(i.shape) for i in xyzs]
    for n,i in enumerate(xyzs):
        coord[n][:] = i[:]
    pe_grad = [np.zeros(i.shape) for i in xyzs]
    path_grad = [np.zeros(i.shape) for i in xyzs]
    massm = genMassMat(atom)
    energy[0] = efirst
    energy[-1] = elast
    for ncycle in range(1, maxcycle+1):
        print("Cycle %i"%ncycle)
        # calc gradient of PE
        for nimage in range(1,len(xyzs)-1):
            e,g = calcGauGrad(atom, coord[nimage], template)
            energy[nimage] = e
            pe_grad[nimage] = g 
            print("E: %.4f  "%e, end="")
        print()
        # get pathway
        alpha = [0.0]
        for n in range(1,len(xyzs)):
            alpha.append(sum(alpha) + distance(xyzs[n-1], xyzs[n]))
        alpha = [i / sum(alpha) for i in alpha]
        posfunc = genPosFunc(coord, alpha)
        if "NEB" in method:
            # get direction of NEB gradient
            for n,a in enumerate(alpha):
                if n == 0 or n == len(alpha):
                    continue
                path_grad[n] = numGrad(posfunc, a)
            # calc elastic band grad
            neb_grad = [np.zeros(i.shape) for i in xyzs]
            for i in range(1,len(xyzs)-1):
                k1 = spring(coord[i], coord[i-1], Kspring)
                k2 = spring(coord[i], coord[i+1], Kspring)
                neb_grad[i] = k1 + k2
            # summarize PE grad and NEB grad
            for i in range(1, len(xyzs)-1):
                gv, gh = decomp(pe_grad[i], path_grad[i])
                if "CI" in method:
                    if i == np.argmax(energy):
                        gh = - gh
                else:
                    _, gh = decomp(neb_grad[i], path_grad[i])
                gsum = gv + gh
                # renew position
                coord[i] = coord[i] - Rstep * gsum / massm 
        # output pathway
        for i in range(len(xyzs)):
            title = "Cycle %i    Image %i    Energy:%.6f\n"%(ncycle, i, energy[i] / EH)
            writeXYZ("pathway-%i.xyz"%i, atom, coord[i], title=title, append=True)

