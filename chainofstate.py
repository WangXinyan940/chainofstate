import numpy as np 
from scipy import interpolate
import os,sys
 
class Topo(object): 
 
    def __init__(self,atoms): 
        self.atoms = atoms 
 
 
def readXYZ(fname): 
    with open(fname, "r") as f: 
        text = f.readlines() 
    line = int(text[0].strip()) 
    text = text[2:line+2] 
    text = [i.strip().split() for i in text] 
    topo = Topo([i[0] for i in text]) 
    xyz = [[float(i[1]),float(i[2]),float(i[3])] for i in text] 
    return topo,np.array(xyz) 
 
def writeXYZ(name,topo,xyz,title="Title"):
    with open("%s.xyz"%name, "w") as f:
        anum = len(topo.atoms)
        f.write("%i\n%s\n"%(anum,title))
        for i in range(anum):
            f.write("%s%15.8f%15.8f%15.8f\n"%(topo.atoms[i],xyz[i,0],xyz[i,1],xyz[i,2]))

#XTB
def writeXTBForce(topo, xyz,n): 
    with open("force.xyz", "w") as f: 
        anum = len(topo.atoms) 
        f.write("%i\nForce\n"%anum) 
        for i in range(anum): 
            f.write("%s%15.8f%15.8f%15.8f\n"%(topo.atoms[i],xyz[i,0],xyz[i,1],xyz[i,2])) 
 
def readXTBForce(): 
    with open("grad", "r") as f: 
        text = f.readlines() 
    text = [float(i[4:].strip().split()[0]) if "*****" not in i else 0.0 for i in text if len(i.strip()) > 10] 
    text = np.array(text) 
    return text.reshape((int(len(text) / 3), 3)) 
 
def readXTBEnergy(): 
    with open("LOG", "r") as f: 
        text = f.readlines() 
    text = [i for i in text if "total E" in i][0] 
    return float(text.strip().split()[-1]) 

def mkXTBFunc(command):
    tot_command = "%s force.xyz -chrg 0 -uhf 0 -grad > LOG"%command
    def calcXTBForce(topo, xyz,n): 
        writeXTBForce(topo, xyz,n) 
        os.system(tot_command) 
        energy = readXTBEnergy() 
        force = readXTBForce()
        os.system("rm LOG force.xyz energy grad") 
        return energy,force 
    return calcXTBForce

def writeXTBHess(topo, xyz): 
    with open("hess.xyz", "w") as f: 
        anum = len(topo.atoms) 
        f.write("%i\nHessian\n"%anum) 
        for i in range(anum): 
            f.write("%s%15.8f%15.8f%15.8f\n"%(topo.atoms[i],xyz[i,0],xyz[i,1],xyz[i,2])) 

def calcXTBHess(topo,xyz):
    writeXTBHess(topo, xyz)
    os.system(os.environ["XTB_COMMAND"] + "hess.xyz -chrg 0 -uhf 0 -grad > HESSLOG")
    with open("hessian", "r") as f:
        text = f.readlines()
    text = [[float(j) for j in i.strip().split()] for i in text[1:]]
    hess = []
    for i in text:
        for j in i:
            hess.append(j)
    hess = np.array(hess).reshape((xyz.shape[0] * xyz.shape[1],xyz.shape[0] * xyz.shape[1]))
    return hess

#Gaussian
def writeGauForce(topo, xyz, n):
    with open("tmp", "w") as f:
        anum = len(topo.atoms)
        for i in range(anum):
            f.write("%s%15.8f%15.8f%15.8f\n"%(topo.atoms[i],xyz[i,0],xyz[i,1],xyz[i,2]))
    if "force_%i.chk"%n in os.listdir("."):
        os.system('echo "%oldchk=force_{}.chk\n%chk=force.chk" | cat - template.gjf tmp tmpend > force.gjf && rm tmp'.format(n))
    else:
        os.system("echo %chk=force.chk | cat - template0.gjf tmp tmpend > force.gjf && rm tmp")

def readGauEnerForce(topo):
    with open("force.log", "r") as f:
        text = f.readlines()
        ener = [i for i in text if "SCF Done:" in i]
        if len(ener) != 0:
            ener = ener[-1]
            ener = np.float64(ener.split()[4])
        else:
           ener = np.float64([i for i in text if "Energy=" in i][-1].split()[1]) 
        for ni, li in enumerate(text):
            if "Forces (Hartrees/Bohr)" in li:
                break
        forces = text[ni+3:ni+3+len(topo.atoms)]
        forces = [i.strip().split()[-3:] for i in forces]
        forces = [[np.float64(i[0]), np.float64(i[1]), np.float64(i[2])] for i in forces]
    return ener,-np.array(forces)

def mkGauFunc(command):
    def calcGauForce(topo,xyz,n):
        writeGauForce(topo, xyz, n)
        os.system("g09 force.gjf")
        e,f = readGauEnerForce(topo)
        os.system("cp force.chk force_%i.chk"%n)
        os.system("rm force.gjf force.log")
        return e,f
    return calcGauForce

#ORCA
def writeORCAForce(topo,xyz):
    with open("tmp", "w") as f:
        anum = len(topo.atoms)
        for i in range(anum):
            f.write("%s%15.8f%15.8f%15.8f\n"%(topo.atoms[i],xyz[i,0],xyz[i,1],xyz[i,2]))
        f.write("*\n\n\n\n")
    os.system("cat template.inp tmp > force.inp && rm tmp")

def readORCAEnerForce():
    with open("force.engrad", "r") as f:
        text = f.readlines()
    text = [i.strip() for i in text if i[0] != "#"]
    tnum = int(text[0])
    e = float(text[1])
    f = [float(i) for i in text[2:2+tnum*3]]
    f = np.array(f).reshape((tnum,3))
    return e,f

def mkORCAFunc(command):
    def calcORCAForce(topo,xyz,n):
        writeORCAForce(topo,xyz)
        if n > 0:
            os.system("cp force_%i.gbw force.gbw"%(n))
        os.system(command+' force.inp "--map-by socket:OVERSUBSCRIBE" > LOG')
        e,f = readORCAEnerForce()
        os.system("cp force.gbw force_%i.gbw"%n)
        os.system("rm force.*")
        return e,f
    return calcORCAForce

def testForce(topo,xyz):
    force = np.zeros(xyz.shape)
    x,y = xyz[0,0],xyz[0,1]
    func1 = lambda x,y: np.exp(- (x - 5) ** 2 / 2 / 16) * np.exp(- (y - 5) ** 2 / 2 / 16)
    func2 = lambda x,y: np.exp(- (x + 5) ** 2 / 2 / 16) * np.exp(- (y + 5) ** 2 / 2 / 16)
    func3 = lambda x,y: 0.00025 * x ** 2 * y ** 2
    e = func1(x,y) + func2(x,y) + func3(x,y)
    force[0,0] = 0.0005*x*y**2 + (-x/16 - 5/16)*np.exp(-(x + 5)**2/32)*np.exp(-(y + 5)**2/32) + (-x/16 + 5/16)*np.exp(-(x - 5)**2/32)*np.exp(-(y - 5)**2/32)
    force[0,1] = 0.0005*x**2*y + (-y/16 - 5/16)*np.exp(-(x + 5)**2/32)*np.exp(-(y + 5)**2/32) + (-y/16 + 5/16)*np.exp(-(x - 5)**2/32)*np.exp(-(y - 5)**2/32)
    return e,force


#Chain-of-state
def distance(a,b): 
    return (((a - b) ** 2).sum() / a.shape[0]) ** 0.5 
 
def decomp(a,b):
    #return va,vb
    #va + vb = a
    #va .* vb = 0
    #vb parrallel to b
    m = np.dot(a,b) / np.dot(b,b)
    return a - m * b, m * b


def spring(a,b,sk=1.0): 
    dist = (((a - b) ** 2).sum() / a.shape[0]) ** 0.5 
    fa, fb = np.zeros(a.shape), np.zeros(b.shape) 
    for ia in range(a.shape[0]): 
        fa[ia] = (a[ia] - b[ia]) / dist 
    fa = -sk * fa 
    fb[:] = -fa[:] 
    return fa,fb  

def cispring(a,b,ei,emax,eref,deltak=0.00125,kmax=1.0):
    if ei > eref:
        k = kmax - deltak * (emax - ei) / (emax - eref)
    else:
        k = kmax - deltak
    dist = (((a - b) ** 2).sum() / a.shape[0]) ** 0.5
    fa, fb = np.zeros(a.shape), np.zeros(b.shape)
    for ia in range(a.shape[0]):
        fa[ia] = (a[ia] - b[ia]) / dist
    fa = -k * fa
    fb[:] = -fa[:]
    return fa,fb

def score(i,grad):
    t = np.sqrt((grad[i] ** 2).sum() / grad[i].shape[0])
    return 1. / (1. + np.exp(-(t + 5)))

def scale(force,k):
    rf = force.reshape((int(force.shape[0]/3),3)) * 0.5
    maxd = (rf ** 2).sum(axis=1).max() ** 0.5
    if maxd < k:
        return force * 0.5
    else:
        return force * 0.5 / maxd * k

def runChainOfState(topo,xyzs,force,method="STRING",fixend=False,k=0.1,sk=0.5,delta=0.00001,enfirst=0.0,enlast=0.0):
    calcs = []
    for n,xyz in enumerate(xyzs):
        if fixend:
            if n == 0 or n == len(xyzs) - 1:
                calcs.append((enfirst if n == 0 else enlast,np.zeros(xyz.shape)))
                print(calcs[-1][0], end=" ")
                continue
        calcs.append(force(topo,xyz,n))
        print(calcs[-1][0], end=" ")
    energies = [i[0] for i in calcs]
    if method == "CINEB" and fixend:
        energies[-1] = enlast
        energies[0] = enfirst
    energies = np.array(energies)
    print()
    forces = [i[1] * 1.889725989 for i in calcs] 
    forces = [i - i.mean(axis=0) for i in forces]
    forces = [i.reshape((i.shape[0]*i.shape[1],)) for i in forces]
    prexyzs = [i.reshape((i.shape[0]*i.shape[1],)) for i in xyzs]

    ss = [0.0] 
    for i in range(len(prexyzs)-1): 
        ss.append(distance(prexyzs[i],prexyzs[i+1])) 
    alpha = [sum(ss[:i+1])/sum(ss) for i in range(len(ss))]
    grad = [np.zeros(i.shape) for i in prexyzs]
    for d in range(prexyzs[0].shape[0]):
        y = [i[d] for i in prexyzs]
        func = interpolate.interp1d(alpha,y,kind="cubic")
        for n in range(len(prexyzs)):
            if n == 0:
                grad[n][d] = (func(delta) - prexyzs[n][d]) / delta
            elif n == len(prexyzs) - 1:
                grad[n][d] = (prexyzs[n][d] - func(1.0 - delta)) / delta
            else:
                grad[n][d] = (func(alpha[n] + delta) - func(alpha[n] - delta)) / 2.0 / delta
    #force decomposition
    if method == "STRING":
        newforces = []
        for i in range(len(forces)):
            vrt,prl = decomp(forces[i],grad[i])
            newforces.append(vrt)
        #renew position
        if fixend:
            newforces[0][:] = 0
            newforces[-1][:] = 0
        else:
            newforces[0][:] = forces[0][:]
            newforces[-1][:] = forces[-1][:]
        newforces = [scale(i,k) for i in newforces]
        newxyzs = [prexyzs[i] - newforces[i] for i in range(len(prexyzs))]
        #interpolation again
        ss = [0.0] 
        for i in range(len(newxyzs)-1): 
            ss.append(distance(newxyzs[i],newxyzs[i+1])) 
        alpha = np.array([sum(ss[:i+1])/sum(ss) for i in range(len(ss))])
        res = np.zeros((len(newxyzs),newxyzs[0].shape[0]))
        for d in range(newxyzs[0].shape[0]):
            func = interpolate.interp1d(alpha,np.array([i[d] for i in newxyzs]),kind="cubic")
            tmp = func(np.linspace(0.0,1.0,len(ss)))
            for n in range(len(tmp)):
                res[n,d] = tmp[n]
        #return
        return energies,[i.reshape(xyzs[0].shape) for i in res] 
    if method == "SIMPLESTRING":
        newforces = [i for i in forces]
        #renew position
        if fixend:
            newforces[0][:] = 0
            newforces[-1][:] = 0
        else:
            newforces[0][:] = forces[0][:]
            newforces[-1][:] = forces[-1][:]
        newforces = [scale(i,k) for i in newforces]
        newxyzs = [prexyzs[i] - newforces[i] for i in range(len(prexyzs))]
        #interpolation again
        ss = [0.0] 
        for i in range(len(newxyzs)-1): 
            ss.append(distance(newxyzs[i],newxyzs[i+1])) 
        alpha = np.array([sum(ss[:i+1])/sum(ss) for i in range(len(ss))])
        res = np.zeros((len(newxyzs),newxyzs[0].shape[0]))
        for d in range(newxyzs[0].shape[0]):
            func = interpolate.interp1d(alpha,np.array([i[d] for i in newxyzs]),kind="cubic")
            tmp = func(np.linspace(0.0,1.0,len(ss)))
            for n in range(len(tmp)):
                res[n,d] = tmp[n]
        #return
        return energies,[i.reshape(xyzs[0].shape) for i in res] 
    elif method == "TSTRING":
        newforces = [i for i in forces]
        #renew position
        if fixend:
            newforces[0][:] = 0
            newforces[-1][:] = 0
        else:
            newforces[0][:] = forces[0][:]
            newforces[-1][:] = forces[-1][:]
        newforces = [scale(i,k) for i in newforces]
        newxyzs = [prexyzs[i] - newforces[i] for i in range(len(prexyzs))]
        #choose sample
        distmat = np.zeros((len(newxyzs), len(newxyzs)))
        for i in range(len(newxyzs)):
            for j in range(i,len(newxyzs)):
                if i == j:
                    distmat[i,j] = 0.0
                else:
                    distmat[i,j] = distance(newxyzs[i].newxyzs[j])
                    distmat[j,i] = distance(newxyzs[i].newxyzs[j])
        
        #interpolation again
        ss = [0.0] 
        for i in range(len(newxyzs)-1): 
            ss.append(distance(newxyzs[i],newxyzs[i+1])) 
        alpha = np.array([sum(ss[:i+1])/sum(ss) for i in range(len(ss))])
        res = np.zeros((len(forces),newxyzs[0].shape[0]))
        for d in range(newxyzs[0].shape[0]):
            func = interpolate.interp1d(alpha,np.array([i[d] for i in newxyzs]),kind="cubic")
            tmp = func(np.linspace(0.0,1.0,len(forces)))
            for n in range(len(tmp)):
                res[n,d] = tmp[n]
        #return
        return energies,[i.reshape(xyzs[0].shape) for i in res]
    elif method == "NEB":
        along,vertical = [np.zeros(i.shape) for i in forces],[]
        for i in range(len(forces)):
            vrt,prl = decomp(forces[i],grad[i])
            vertical.append(vrt)
        for i in range(len(forces)-1):
            ia,ib = i,i+1
            fa,fb = spring(prexyzs[ia],prexyzs[ib],sk)
            vrt,prl = decomp(fa,grad[ia])
            along[ia] = prl
            vrt,prl = decomp(fa,grad[ib])
            along[ib] = prl
        newforces = [vertical[i] + along[i] for i in range(len(vertical))]
        newforces[0][:] = forces[0][:]
        newforces[-1][:] = forces[-1][:]
    elif method == "CINEB":
        ts_images = []
        ts_prl = []
        for n,i in enumerate(energies):
            if n == 0 or n == len(energies) - 1:
                continue
            if n == 1:
                if i > max([energies[0],energies[2],energies[3]]):
                    ts_images.append(n)
                continue
            if n == len(energies) - 2:
                if i > max([energies[n+1],energies[n-1],energies[n-2]]):
                    ts_image.append(n)
                continue
            if i >= max([energies[n-1],energies[n-2],energies[n+1],energies[n+2]]):
                ts_images.append(n)
                continue
        along,vertical = [np.zeros(i.shape) for i in forces],[]
        for i in range(len(forces)):
            vrt,prl = decomp(forces[i],grad[i])
            vertical.append(vrt)
            if i in ts_images:
                ts_prl.append(prl)

        emax = max(energies)
        eref = min([energies[0],energies[-1]])

        for i in range(len(forces)-1):
            ia,ib = i,i+1
            fa,fb = cispring(prexyzs[ia],prexyzs[ib],max([energies[ia],energies[ib]]),emax,eref,kmax=sk,deltak=0.5*sk)
            vrt,prl = decomp(fa,grad[ia])
            along[ia] += prl
            vrt,prl = decomp(fa,grad[ib])
            along[ib] += prl
        for n in range(len(ts_images)):
            along[ts_images[n]] = - 2 * ts_prl[n]
        newforces = [vertical[i] + along[i] for i in range(len(vertical))]
        newforces[0][:] = forces[0][:]
        newforces[-1][:] = forces[-1][:]
    #renew position
    newforces = [scale(i,k) for i in newforces]
    if fixend:
        newforces[0][:] = 0
        newforces[-1][:] = 0
    newxyzs = [prexyzs[i] - newforces[i] for i in range(len(prexyzs))]
    #return
    return energies,[i.reshape(xyzs[0].shape) for i in newxyzs]    




if __name__ == "__main__":
    #Argparse
    import argparse
    parser = argparse.ArgumentParser()
    #QM Engine
    parser.add_argument("-e","--engine",help="QM Engine [G09/XTB/ORCA]",default="G09")
    #Algorithm
    parser.add_argument("-a","--alg",help="Algorithm [STRING/NEB/CINEB]",default="STRING")
    #Fix-end
    parser.add_argument("-f","--fixend",help="Fix-end [Default False]", action="store_true")
    #k for norm of forces
    parser.add_argument("-n","--norm",help="Norm const for displacement [Default 0.1 A]", type=float,default=0.1)
    #sk for spring K const
    parser.add_argument("-k","--springK",help="Spring const [Default 0.0025]", type=float,default=0.0025)
    #Structures
    parser.add_argument("--inter",help="Structures added between two init structures. [Default 3]", type=int,default=3)
    #Files
    parser.add_argument("files",metavar="F",type=str,nargs="+",help="File names of init structures.")
    #Cycles
    parser.add_argument("-c","--cycles",type=int,help="Iteration cycles [Default 20]", default=20)
    #EnStart
    parser.add_argument("--enfirst", type=float, help="Energy of the first image. Only be used in CINEB. [kcal/mol]", default=-9999.0)
    #EnEnd
    parser.add_argument("--enlast", type=float, help="Energy of the last image. Only be used in CINEB. [kcal/mol]", default=-9999.0)
    print("###################################################")
    print("#                                                 #")
    print("# Chain of State Toolbox                          #")
    print("#                                                 #")
    print("# ENV needed:                                     #")
    print("#                                                 #")
    print("# ORCA: ORCA_COMMAND                              #")
    print("# XTB: XTB_COMMAND                                #")
    print("#                                                 #")
    print("###################################################")
    args = parser.parse_args()
    env_dir = os.environ

    inter = args.inter + 1
    initxyz = []
    for i in args.files:
        topo,xyz = readXYZ(i)
        initxyz.append(xyz)
    xyzs = [initxyz[0]]
    for i in range(len(initxyz)-1):
        for j in range(1,inter):
            xyzs.append(initxyz[i] * (inter - 1 - j) / (inter - 1) + initxyz[i+1] * j / (inter - 1))

    engine = args.engine
    method = args.alg
    if engine == "G09":
        force = mkGauFunc("g09")
    if engine == "XTB":
        force = mkXTBFunc(env_dir["XTB_COMMAND"])
    if engine == "ORCA":
        force = mkORCAFunc(env_dir["ORCA_COMMAND"])

    for i in range(args.cycles):
        print("CYCLE %i:"%i)
        e,xyzs = runChainOfState(topo,xyzs,force,method=method,fixend=args.fixend,k=args.norm,sk=args.springK,enfirst=args.enfirst,enlast=args.enlast)
        for n,xyz in enumerate(xyzs):
            writeXYZ("force%i_%i"%(i,n),topo,xyz,title="%f"%e[n])
        #e,xyzs = gausimplestring(topo,xyzs)
        print("MAX: %f\n"%e[1:-1].max())

