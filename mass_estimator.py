from stellarpop.estimator import Estimator

class MassEstimator(Estimator):
    """
    An object used to determine estimates of stellar masses. This inherits
        from the base class NestedSampler, although this functionality is not
        necessary for simple MCMC chains.
    """

    def __init__(self,priors,data,model,constraints=[]):
        self.data = data
        self.model = model
        self.priors = priors
        self.names = priors.keys()
        self.constraints = constraints

        if 'redshift' not in self.names:
            self.format = 'old'
        else:
            self.format = 'new'

    def fastMCMC(self,niter,nburn,nthin=1):
        from Sampler import SimpleSample as sample
        from scipy import interpolate
        import pymc,numpy,time
        import ndinterp

        models = self.model.models
        data = self.data
        filters = data.keys()

        # Remove redshift dict
        for key in models.keys():
            if type(models[key])==type({}):
                z = models[key].keys()[0]
                models[key] = models[key][z]

        pars = [self.priors[key] for key in self.names]

        ax = {}
        doExp = []
        cube2par = []
        i = 0
        for key in self.model.axes_names:
            ax[key] = i
            i += 1
        i = 0
        for key in self.names:
            if key[0]=='X':
                continue
            if key.find('log')==0:
                pntkey = key.split('log')[1]
                #self.priors[key].value = numpy.log10(best[ax[pntkey]])
                doExp.append(True)
            else:
                pntkey = key
                doExp.append(False)
                #self.priors[key].value = best[ax[pntkey]]
            cube2par.append(ax[pntkey])
        doExp = numpy.array(doExp)==True
        par2cube = numpy.argsort(cube2par)

        # add stellar mass parameters
        pars.append(pymc.Uniform('log_Mlens',9.,12.))
        pars.append(pymc.Uniform('log_Msrc',9.,12.))

        M = numpy.empty(len(filters))
        D = numpy.empty(len(filters))
        V = numpy.empty(len(filters))
        for i in range(D.size):
            f = filters[i]
            D[i] = data[f]['mag']
            V[i] = data[f]['sigma']**2
        @pymc.deterministic
        def mass_and_logp(value=0.,pars=pars):
            logp = 0
            p = numpy.array(pars[:-2])
            p[doExp] = 10**p[doExp]
            p = numpy.atleast_2d(p[par2cube])
            mlens, msrc = pars[-2],pars[-1]
            for i in range(M.size):
                filt = filters[i]
                M[i] = models[filt].eval(p)
                if M[i]==0:
                    return [-1.,-1e300]
                if len(D[i])==1:
                    # sdss magnitude
                    ml,ms = M[i][0] - 2.5*mlens, M[i][1] - 2.5*Msrc
                    f = 10**(-0.4*ml) + 10**(-0.4*ms)
                    f = -2.5*np.log10(f)
                    logp += -0.5*(f-D[i])**2./V[i]
                elif len(D[i]) ==2:
                    # HST/Keck magnitude
                    logp += -0.5*(M[i][0] - 2.5*mlens - D[i][0])**2./V[i][0] - 0.5*(M[i][1] - 2.5*msrc - D[i][1])**2./V[i][1]
            return logp

        @pymc.observed
        def loglikelihood(value=0.,lp=mass_and_logp):
            return lp

        cov = []
        for key in self.names:
            if key=='age':
                cov.append(0.5)
            elif key=='logage':
                cov.append(0.03)
            elif key=='tau':
                cov.append(0.1)
            elif key=='logtau':
                cov.append(0.03)
            elif key=='tau_V':
                cov.append(self.priors[key]['prior'].value/20.)
            elif key=='logtau_V':
                cov.append(0.1)
            elif key=='tauV':
                cov.append(self.priors[key]['prior'].value/20.)
            elif key=='logtauV':
                cov.append(0.1)
            elif key=='Z':
                cov.append(self.priors[key]['prior'].value/20.)
            elif key=='logZ':
                cov.append(0.03)
            elif key=='redshift':
                P = self.priors['redshift']
                if type(P)==type(pymc.Normal('t',0.,1)):
                    cov.append(P.parents['tau']**-0.5)
                elif type(P)==type(pymc.Uniform('t',0.,1.)):
                    cov.append((P.parents['upper']-P.parents['lower'])/10.)
                else:
                    cov.append(P.parents['cov'])
                #cov.append(0.1)
        cov += [0.5,0.5] # masses
        cov = numpy.array(cov)

        costs = self.constraints+[loglikelihood]
        from SampleOpt import Sampler,AMAOpt
        S = AMAOpt(pars,costs,[mass_and_logp],cov=cov)
        S.sample(nburn/4)

        S = Sampler(pars,costs,[mass_and_logp])
        S.setCov(cov)
        S.sample(nburn/4)

        S = Sampler(pars,costs,[mass_and_logp])
        S.setCov(cov)
        S.sample(nburn/2)

        logps,trace,dets = S.result()
        cov = numpy.cov(trace[nburn/4:].T)

        S = AMAOpt(pars,costs,[mass_and_logp],cov=cov/4.)
        S.sample(nburn/2)
        logps,trace,dets = S.result()

        S = Sampler(pars,costs,[mass_and_logp])
        S.setCov(cov)
        S.sample(nburn/2)

        logps,trace,dets = S.result()
        cov = numpy.cov(trace[nburn/4:].T)

        S = Sampler(pars,costs,[mass_and_logp])
        S.setCov(cov)
        S.sample(niter)

        logps,trace,dets = S.result()
        logL = dets['mass_and_logp'].T
        o = {'logP':logps,'logL':logL}
        cnt = 0
        for key in self.names:
            o[key] = trace[:,cnt].copy()
            cnt += 1
        return o
