import pymc,sys,cPickle
from stellarpop import distances,mass_estimator
from math import log10
from numpy import loadtxt
import numpy,glob

modelname = sys.argv[1]
f = open(modelname,'rb')
model = cPickle.load(f)
f.close()

redshift = model.redshifts[0]
dist = distances.Distance()
dist.OMEGA_M = 0.3
dist.OMEGA_L = 0.7
dist.h = 0.7

t_univ = dist.age(redshift)
if t_univ>13.5:
    t_univ = 13.5
t_univ = t_univ-dist.age(10.)

tstart = dist.age(redshift)-dist.age(redshift+0.1)
tend = dist.age(redshift)-dist.age(5.)

""" Create the priors dictionary """
priors = {}


""" Z PRIOR """
priors['logZ'] = {}
priors['logZ'] = pymc.Uniform('logZ',-4.,log10(0.05))


""" AGE PRIOR """
priors['age'] = {}
priors['age'] = pymc.Uniform('age',tstart,tend)


""" TAU PRIOR """
priors['tau'] = {}
priors['tau']['prior'] = pymc.Uniform('tau',0.04,5.1)


""" TAU_V PRIOR """
priors['logtau_V'] = {}
priors['logtau_V'] = pymc.Uniform('logtau_V',-2.,log10(2.))

""" MASS PRIORs - no longer linear  """
#priors['log_Mlens'] = {}
#priors['log_Mlens'] = pymc.Uniform('log_Mlens',9.,12.)
#priors['log_Msrc'] = {}
#priors['log_Msrc'] = pymc.Uniform('log_Msrc',9.,12.)

data = open('src_photometry.cat').readlines()
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
# load up our data.  
# this needs to be my catalogue. Add in sdss photometry. And DUST CORRECT HERE.
for line in data:
    name,g,r,i,z,dg,dr,di,dz,Klens,Ksrc,dKlens,dKsrc,Vlens,Vsrc,dVlens,dVsrc,Ilens,Isrc,dIlens,dIsrc = line.split()
    if int(name)!=int(modelname[4:7]):
        continue
    d = {}
    d['g_SDSS'] = {'mag':g,'sigma':dg}
    d['r_SDSS'] = {'mag':r,'sigma':dr}
    d['i_SDSS'] = {'mag':i,'sigma':di}
    d['z_SDSS'] = {'mag':z,'sigma':dz}
    d['Kp_NIRC2'] = {'mag':[Klens,Ksrc],'sigma':[dKlens,dKsrc]}
    if bands[name] == 'F555W':
        d['F555W_ACS'] = {'mag':[Vlens,Vsrc],'sigma':[dVlens,dVsrc]}
    elif bands[name] == 'F606W':
        d['F606W_ACS'] = {'mag':[Vlens,Vsrc],'sigma':[dVlens,dVsrc]}
    else:
        print 'ERROR'
    d['F814W_ACS'] = {'mag':[Ilens,Isrc],'sigma':[dIlens,dIsrc]}
    sampler = mass_estimator.MassEstimator(priors,data,model)
    break

sampler = mass_estimator.MassEstimator(priors,d,model)
result = sampler.fastMCMC(20000,10000)
print "%03s  %5.2f %4.2f"%(name,result['logmass'].mean(),result['logmass'].std())
of = open(modelname.replace('chabBC03.model','inference.dat'),'wb')
cPickle.dump(result,of,2)
of.close()
