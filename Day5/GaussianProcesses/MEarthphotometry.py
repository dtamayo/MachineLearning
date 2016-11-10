'''
Compute the full light curve for GJ 1132 using the MEarth photometry supplied 
by Zach.
'''
from imports import *
import lcmodel

class MEarthphotometry:
    
    def __init__(self, outsuffix='', Prot=122., thetagp=None,
                 GPonly=True, Rs=.207, T0=2457184.55786):
        '''GPonly = boolean indicating whether or not to model the light
        curve as a QP GP or a GP (prbly a squared exponential) plus a 
        sine wave.''' 

        # Add obvious stuff
        self.outsuffix = outsuffix
        self.Prot = Prot
        self.Rs = Rs
	self.T0 = T0
        self.GPonly = GPonly
        if thetagp == None:
            thetagp = np.array((1e-5,4e4,1e-1,127,.02,.02,125))
        else:
            thetagp = thetagp
        if self.GPonly:
            self.thetagp = np.log(thetagp[:4])
        else:
            thetagp = thetagp[np.array((0,1,4,5,6))]
            thetagp[:2] = np.log(thetagp[:2])
            self.thetagp = thetagp

        # Get MEarth photometric data
	fname = glob.glob('data/2MASS*')
        d = np.loadtxt(fname[0])
        self.bjd = d[:,0]
        self.mag = d[:,18]  # corrected differential magnitude
        self.emag = d[:,2]
        self.mag0 = d[:,4]  # zero-point mag (mag0 < -0.5 are suspicious)
        self.CM = d[:,17]   # common-mode; c(t)
        
        #self.bjdbin()
        self.trimnbin()
        
    
    ###########################################################################
    ## Trim and bin BJD array
    ###########################################################################
    def bjdbin(self, binwidth=4):
        '''Bin a the timeseries in time with bins of size binwidth in 
        days.'''
        bins = np.arange(min(self.bjd), max(self.bjd), binwidth)
        nbin = bins.size
        dig = np.digitize(self.bjd, bins)
        magbin = np.array([np.mean(self.mag[dig == i]) 
                           for i in range(1, nbin)])
        emagbin = np.array([np.std(self.mag[dig == i]) 
                            for i in range(1, nbin)])
        bjdbin = np.array([np.mean([bins[i],bins[i+1]]) 
                           for i in range(nbin-1)])
        good = np.where(np.isfinite(magbin))[0]
        self.bjdbin = bjdbin[good]
        self.magbin = magbin[good]        
        self.emagbin = emagbin[good]
    

    def trimnbin(self, binwidth=4):
        '''Trim the timeseries to only include measurements with 
        mag0 > -0.5
        Rebin the leftover data'''
        # Remove bad measurements
        good = np.where(self.mag0 > -.5)[0]
        self.bjdtrim = self.bjd[good]
        self.magtrim = self.mag[good]
        self.emagtrim = self.emag[good]
        
        # Bin the new data
        bins = np.arange(min(self.bjdtrim), max(self.bjdtrim), binwidth)
        nbin = bins.size
        dig = np.digitize(self.bjdtrim, bins)
        magbin = np.array([np.mean(self.magtrim[dig == i]) 
                           for i in range(1,nbin)])
        emagbin = np.array([np.std(self.magtrim[dig == i]) 
                            for i in range(1,nbin)])
        bjdbin = np.array([np.mean([bins[i],bins[i+1]]) 
                           for i in range(nbin-1)])
        good = np.where(np.isfinite(magbin))[0]
        self.bjdtrimbin  = bjdbin[good]
        self.magtrimbin  = magbin[good]
        self.emagtrimbin = emagbin[good]

    ###########################################################################
    ## Lomb-Scargle periodogram
    ###########################################################################
    def compute_periodogram(self):
        '''Compute the Lomb-Scargle periodogram for the light curve 
        using the Systemic software.'''
        # Remove GJ1132b signal and save data
        forsystemic = np.zeros((self.bjdtrim.size, 3))
        forsystemic[:,0] = self.bjdtrim
        forsystemic[:,1] = self.magtrim
        forsystemic[:,2] = self.emagtrim
        np.savetxt('/home/ryan/linux/Systemic/datafiles/MEarth.vels',
                   forsystemic, delimiter='\t', fmt='%.6f')

        # Run R script to compute periodogram and save data to a file
        t0 = time.time()
        os.system('Rscript periodogram.r')
        print 'Computing the periodogram took %.3e minutes.'%((time.time()-t0)/60.)

        # Get periodogram data
        d = np.loadtxt('periodogram.dat')
        self.periodogramP    = d[:,0]
        self.periodogrampow  = d[:,1]
        self.periodogramfaps = d[:,2]
        d = np.loadtxt('peaks.dat')
        self.peaksP    = d[:,0]
        self.peaksfaps = d[:,1]
        # Remove large unwanted files
        os.system('rm periodogram.dat')
        os.system('rm peaks.dat')

    ##########################################################################
    ## Model the RV variation due to spots form photometry 
    ##########################################################################
    def estimate_RVpert(self, Tstar=3270., dTspot=600, VsinI=2.):
        '''Estimate the radial velocity perturbation (m/s) from starspots 
        from the star's light curve which is assumed to be due to the 
        rotation of said spots. The simplified conversion equation is
        Eq 10 in Dumusque et al 2014 (SOAP paper).
        
        Tstar = effective T of GJ1132 (K)
        dTspot = T difference between the stellar photosphere and the 
        starspot.
        VsinI = projected stellar rotation velocity of GJ1132 (actually an 
        upper limit from Berta-Thompson etal 2015)'''
        # Compute the fractional starspot coverage; F(t)
        wlnm = 700.
        Bstar = planck(Tstar, lam=wlnm*1e-9)
        Bspot = planck(Tstar-dTspot, lam=wlnm*1e-9)
        self.fluxtrimbin = 10**(-.4*self.magtrimbin)
        self.Ftrimbin = Bstar/(Bstar-Bspot) * (1-self.fluxtrimbin)

        # Use idealized Eq 10 to compute RV perturbation from spots (m/s)
        self.dRVtrimbin = self.Ftrimbin * (1.83 + 9.52*VsinI + 
                                           .79*VsinI**2 - .04*VsinI**3)


    def estimate_RVpert2(self, delta=0, I=90, dVc=None, kappa=None):
        '''Estimate the RV perturbation (m/s) from rotating star spots in a 
        simplfied model from Aigrain et al 2012. The rotational effect 
        Eq 4 and the convective blueshift suppression Eq 5 are included.
        
        delta = assumed spot latitude (deg)
        I = inclination of the star wrt the line-of-sight (deg)
        dVc = difference in velocity of starspot and photosphere (m/s) 
              (default: estimated from a SOAP simulation)
        kappa = surface area ratio of unspotted photosphere to starspot
                (default: estimated from SOAP simulation)
        '''
        # Get the RV perturbation due to flux suppression (m/s)
        fluxmean = 10**(-.4*self.model)
        flux3sig = 10**(-.4*(self.model+3*self.modelerr))
        fluxm3sig = 10**(-.4*(self.model-3*self.modelerr))
        fluxpredmean = 10**(-.4*self.predmodel)
        fluxpred1sig = 10**(-.4*(self.predmodel+self.predmodelerr))
        Veq = 2*np.pi*self.Rs*695500e3/(self.Prot*24*60*60)  # m/s
        # Convert to phase with an approx T0 (i.e. where mag == 0)
        T0 = self.modelbjd[np.abs(self.model) == \
                           np.min(np.abs(self.model))]
        phi = 2*np.pi*(self.modelbjd-T0)/self.Prot 
        phi += .1127  # alignment
        phipred = 2*np.pi*(self.predbjd-T0)/self.Prot
        phipred += .1127
        self.dRV_rotmean = -(1-fluxmean) * Veq * np.cos(np.deg2rad(delta)) * \
                           np.sin(phi) * np.sin(np.deg2rad(I))
        self.dRV_rot3sig = -(1-flux3sig) * Veq * np.cos(np.deg2rad(delta)) * \
                           np.sin(phi) * np.sin(np.deg2rad(I))
        self.dRV_rotm3sig = -(1-fluxm3sig) * Veq * np.cos(np.deg2rad(delta)) * \
                            np.sin(phi) * np.sin(np.deg2rad(I))
        self.dRV_rotpredmean = -(1-fluxpredmean) * Veq * np.cos(np.deg2rad(delta)) * \
                               np.sin(phipred) * np.sin(np.deg2rad(I))
        self.dRV_rotpred1sig = -(1-fluxpred1sig) * Veq * np.cos(np.deg2rad(delta)) * \
                               np.sin(phipred) * np.sin(np.deg2rad(I))

        # Get dVc and kappa from SOAP simulation of one central starspot
        if dVc == None:
            # Get files
            fnames = glob.glob('/home/ryan/Research/SOAP_2/outputs/CCF_PROT'+ \
                               '=125.00_i=90.00_lon=(180.0,0.0,0.0,0.0)_lat'+ \
                               '=(0.0,0.0,0.0,0.0)_size=(0.1000,0.0000,0.00'+ \
                               '00,0.0000)/fits/*fits')
            nfiles = len(fnames)
            RV_BC = np.zeros(nfiles)
            for i in range(nfiles):
                hdr = pyfits.getheader(fnames[i])
                RV_BC[i] = hdr.get('RV_BC')
            self.dVc = np.max(np.abs(RV_BC))*1e3  # m/s
            self.kappa = 1./hdr.get('SIZE1')

        # Get the RV perturbation due to suppresion of CBS
        cosB = np.cos(phi) * np.cos(np.deg2rad(delta)) * \
               np.sin(np.deg2rad(I)) + np.sin(np.deg2rad(delta)) * \
               np.cos(np.deg2rad(I))
        cosBpred = np.cos(phipred) * np.cos(np.deg2rad(delta)) * \
                   np.sin(np.deg2rad(I)) + np.sin(np.deg2rad(delta)) * \
                   np.cos(np.deg2rad(I))
        self.dRV_cmean = (1-fluxmean) * self.dVc * self.kappa * cosB
        self.dRV_c3sig = (1-flux3sig) * self.dVc * self.kappa * cosB
        self.dRV_cm3sig = (1-fluxm3sig) * self.dVc * self.kappa * cosB
        self.dRV_cpredmean = (1-fluxpredmean) * self.dVc * self.kappa * cosBpred
        self.dRV_cpred1sig = (1-fluxpred1sig) * self.dVc * self.kappa * cosBpred
        self.dRVtotmean = self.dRV_rotmean + self.dRV_cmean
        self.dRVtot3sig = self.dRV_rot3sig + self.dRV_c3sig
        self.dRVtotm3sig = self.dRV_rotm3sig + self.dRV_cm3sig
        self.dRVtotpredmean = self.dRV_rotpredmean + self.dRV_cpredmean
        self.dRVtotpred1sig = self.dRV_rotpred1sig + self.dRV_cpred1sig


    def estimate_RVpert3(self, dVc=None, kappa=None):
        '''Estimate the RV perturbation (m/s) from rotating star spots in 
        the FF' model from Aigrain et al 2012. Eqs 10 and 12.

        dVc = difference in velocity of starspot and photosphere (m/s) 
              (default: estimated from a SOAP simulation)
        kappa = surface area ratio of unspotted photosphere to starspot
                (default: estimated from SOAP simulation)
        '''
        # Get the RV perturbation due to flux suppression (m/s)
        self.fullmodel = np.append(self.model, self.predmodel)
        self.fullbjd = np.append(self.modelbjd, self.predbjd)
        self.fullmodelerr = np.append(self.modelerr, self.predmodelerr)
        from uncertainties import unumpy
        ufluxfull = 10**(-.4*unumpy.uarray(self.fullmodel, self.fullmodelerr))
        fluxmeanfull = unumpy.nominal_values(ufluxfull)
        fluxsigfull = unumpy.std_devs(ufluxfull)
        uflux = 10**(-.4*unumpy.uarray(self.model, self.modelerr))
        fluxmean = unumpy.nominal_values(uflux)
        fluxsig = unumpy.std_devs(uflux)
        ufluxpred = 10**(-.4*unumpy.uarray(self.predmodel, self.predmodelerr))
        fluxmeanpred = unumpy.nominal_values(ufluxpred)
        fluxsigpred = unumpy.std_devs(ufluxpred)
        
        # Compute dRV_rot using F and F'
        uFfull = 1 - ufluxfull
        Fmeanfull = unumpy.nominal_values(uFfull)
        Fsigfull = unumpy.std_devs(uFfull)
        uF = 1 - uflux
        Fmean = unumpy.nominal_values(uF)
        Fsig = unumpy.std_devs(uF)
        uFpred = 1 - ufluxpred
        Fmeanpred = unumpy.nominal_values(uFpred)
        Fsigpred = unumpy.std_devs(uFpred)
        
        # Estimate the error of the derivative
        Fdotfull = np.gradient(Fmeanfull)/(24*60*60)
        Fdot = np.gradient(Fmean)/(24*60*60)  # F/sec
        Fdotpred = np.gradient(Fmeanpred)/(24*60*60)  # F/sec
        Psi0 = 1.
        Thetamin = np.min(fluxmean)
        Thetaminpred = np.min(fluxmeanpred)
        f = (Psi0 - Thetamin) / Psi0
        fpred = (Psi0 - Thetaminpred) / Psi0
        Rstar = self.Rs*695500e3

        dRV_rot = -uF*Fdot*Rstar/f
        self.dRV_rotmean = unumpy.nominal_values(dRV_rot)
        self.dRV_rotsig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_rot))), dRV_rot.size)
        dRV_rotpred = -uFpred*Fdotpred*Rstar/fpred
        self.dRV_rotpredmean = unumpy.nominal_values(dRV_rotpred)
        self.dRV_rotpredsig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_rotpred))), dRV_rotpred.size)
        dRV_rotfull = -uFfull*Fdotfull*Rstar/f
        self.dRV_rotfullmean = unumpy.nominal_values(dRV_rotfull)
        self.dRV_rotfullsig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_rotfull))), dRV_rotfull.size)
        '''plt.plot(self.modelbjd, self.dRV_rotmean, 'k-')
        plt.fill_between(self.modelbjd, self.dRV_rotmean+3*self.dRV_rotsig,
                         self.dRV_rotmean-3*self.dRV_rotsig, color='k', alpha=.5)
        plt.plot(self.predbjd, self.dRV_rotpredmean, 'b-')
        plt.fill_between(self.predbjd, self.dRV_rotpredmean+3*self.dRV_rotpredsig,
                         self.dRV_rotpredmean-3*self.dRV_rotpredsig, color='b', alpha=.5)
        plt.show()'''
     
        # Compute dRV_convblueshift using F and some other physical parameters
        if dVc == None:
            # Get files
            fnames = glob.glob('/home/ryan/Research/SOAP_2/outputs/CCF_PROT'+ \
                               '=125.00_i=90.00_lon=(180.0,0.0,0.0,0.0)_lat'+ \
                               '=(0.0,0.0,0.0,0.0)_size=(0.1000,0.0000,0.00'+ \
                               '00,0.0000)/fits/*fits')
            nfiles = len(fnames)
            RV_BC = np.zeros(nfiles)
            for i in range(nfiles):
                hdr = pyfits.getheader(fnames[i])
                RV_BC[i] = hdr.get('RV_BC')
            self.dVc = np.max(np.abs(RV_BC))*1e3  # m/s
            self.kappa = 1./hdr.get('SIZE1')
        else:
            self.dVc = dVc
            self.kappa = kappa
        dRV_c = uF*uF*self.dVc*self.kappa/f
        self.dRV_cmean = unumpy.nominal_values(dRV_c)
        self.dRV_csig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_c))), dRV_c.size)
        dRV_cpred = uFpred*uFpred*self.dVc*self.kappa/fpred
        self.dRV_cpredmean = unumpy.nominal_values(dRV_cpred)
        self.dRV_cpredsig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_cpred))), dRV_cpred.size)
        dRV_cfull = uFfull*uFfull*self.dVc*self.kappa/f
        self.dRV_cfullmean = unumpy.nominal_values(dRV_cfull)
        self.dRV_cfullsig = np.repeat(np.mean(np.abs(unumpy.std_devs(dRV_cfull))), dRV_cfull.size)
        '''plt.plot(self.modelbjd, self.dRV_cmean, 'k-')
        plt.fill_between(self.modelbjd, self.dRV_cmean+3*self.dRV_csig,
                         self.dRV_cmean-3*self.dRV_csig, color='k', alpha=.5)
        plt.plot(self.predbjd, self.dRV_cpredmean, 'b-')
        plt.fill_between(self.predbjd, self.dRV_cpredmean+3*self.dRV_cpredsig,
                         self.dRV_cpredmean-3*self.dRV_cpredsig, color='b', alpha=.5)
        plt.show()'''
               
        # Compute total RV variation
        self.dRV_totmean = self.dRV_rotmean + self.dRV_cmean
        self.dRV_totsig = np.sqrt(self.dRV_rotsig**2 + self.dRV_csig**2)
        self.dRV_totpredmean = self.dRV_rotpredmean + self.dRV_cpredmean
        self.dRV_totpredsig = np.sqrt(self.dRV_rotpredsig**2 + \
                                      self.dRV_cpredsig**2)
        self.dRV_totfullmean = self.dRV_rotfullmean + self.dRV_cfullmean
        self.dRV_totfullsig = np.sqrt(self.dRV_rotfullsig**2 + self.dRV_cfullsig**2)

        '''plt.plot(self.modelbjd, self.dRV_totmean, 'k-')
        plt.fill_between(self.modelbjd, self.dRV_totmean+3*self.dRV_totsig,
                         self.dRV_totmean-3*self.dRV_totsig, color='k', alpha=.5)
        plt.plot(self.predbjd, self.dRV_totpredmean, 'b-')
        plt.fill_between(self.predbjd, self.dRV_totpredmean+3*self.dRV_totpredsig,
                         self.dRV_totpredmean-3*self.dRV_totpredsig, color='b', alpha=.5)
        plt.show()
        plt.close('all')'''



    ##########################################################################
    ## Run MCMC using a QP GP
    ##########################################################################
    def optimize(self, p0=None):
        '''Fit a three-parameter sinusoid model to the light curve.'''
        from scipy.optimize import curve_fit
        def sinemodel(x, As, Ac, Prot):
           return As*np.sin(2*np.pi*x/Prot) + Ac*np.cos(2*np.pi*x/Prot)
        p, c = curve_fit(sinemodel, self.bjdtrim, self.magtrim, 
                         p0=p0)
        self.optmodel = sinemodel(self.bjdtrim, p[0], p[1], p[2])
        self.optresults = p
        self.optresultserr = np.sqrt(np.diag(c))

        
    def runqpgp(self, nsteps=2000, burnin=500, nwalkers=40):
        '''Model the light curve with a QP GP noise model and a 
        sinusoid.'''
        if self.GPonly:
            import GProt as gps 
        else:
            import GPsinerot as gps
        # First pass
        print self.thetagp
        samples,lnprobs,vals=gps.run_emcee_gp(self.thetagp, 
                                              self.bjdtrimbin,
                                              self.magtrimbin, 
                                              self.emagtrimbin,
                                              nsteps=nsteps,
                                              burnin=burnin,
                                              nwalkers=nwalkers)
        self.gpsamples = samples
        self.gplnprobs = lnprobs
        self.gpvals = vals

        # Get best-fit parameters
        if self.GPonly:
            results = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                   zip(*np.percentile(np.exp(self.gpsamples), 
                                                      [16, 50, 84], axis=0))))
            self.hyperparams = results[:,0]
            self.mcmcfullresults = results
        else:
            results = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                   zip(*np.percentile(np.exp(
                                       self.gpsamples[:,:2]), [16, 50, 84], 
                                                      axis=0))))
            self.hyperparams = results[:,0]
            self.mcmcfullresults = results
            results = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                   zip(*np.percentile(self.gpsamples[:,2:], 
                                                      [16, 50, 84], axis=0))))
            self.mcmcparams = results[:,0] 
        '''nparam = self.gpsamples.shape[1]
        params = np.zeros(nparam)
        for i in range(nparam):
            y,x,p = plt.hist(self.gpsamples[:,i], bins=40)
            good = np.where(y == max(y))[0][0]
            params[i] = np.mean((x[good], x[good+1]))
            plt.close('all')
        if self.GPonly:
            self.hyperparams = params
            self.mcmcparams  = None
        else:
            self.hyperparams = params[:2]
            self.mcmcparams  = params[2:]
        print self.thetagp
        print self.hyperparams'''


    def get_model(self):
        '''Compute the QP GP model after running the mcmc.'''
        if self.GPonly:
            # Compute GP model
            a_gp, l_gp, G_gp, P_gp = self.hyperparams
            k1 = kernels.ExpSquaredKernel(l_gp)
            k2 = kernels.ExpSine2Kernel(G_gp, P_gp)
            kernel = a_gp*k1*k2
            gp = george.GP(kernel)
            gp.compute(self.bjdtrimbin, self.emagtrimbin)
            x = np.linspace(min(self.bjd), max(self.bjd), 5e2)
            mu, cov = gp.predict(self.magtrimbin, x)
            std = np.sqrt(np.diag(cov))
            xpred = np.linspace(max(self.bjd), max(self.bjd)+600, 1e3)
            mupred, covpred = gp.predict(self.magtrimbin, xpred)
            stdpred = np.sqrt(np.diag(covpred))   
        else:
            # Compute GP model
            a_gp, l_gp = self.hyperparams
            k1 = kernels.ExpSquaredKernel(l_gp)
            kernel = a_gp*k1
            gp = george.GP(kernel, solver=george.HODLRSolver)
            gp.compute(self.bjdtrimbin, self.emagtrimbin)
            x = np.linspace(min(self.bjd), max(self.bjd), 3e2)
            #mu, cov = gp.predict(self.magtrimbin, x)
            modelsmall = lcmodel.get_lc1(self.mcmcparams, self.bjdtrimbin)
            modellarge = lcmodel.get_lc1(self.mcmcparams, x)
            samples = gp.sample_conditional(self.magtrimbin - modelsmall, 
                                            x, size=100)
            mu = np.mean(samples, axis=0) + modellarge
            std = np.std(samples, axis=0) + modellarge
        self.modelbjd = x
        self.model = mu
        self.modelerr = std
        self.predbjd = xpred
        self.predmodel = mupred
        self.predmodelerr = stdpred


    ##########################################################################
    ## Plotting functions
    ##########################################################################
    def plot_periodogram(self, label=False, pltt=False):
        '''Plot the periodogram along with the of its significant 
        peaks.'''
        plt.figure(figsize=(11,6))
        # Plot power spectrum
        plt.plot(self.periodogramP, self.periodogrampow, 'k-')
        # Highlight the actualy planet period
        plt.axvline(self.Prot, color='b', ls='--')
        # Highlight signficant peaks and their FAPs
        for i in range(5):
            # Get power at the low FAP periods
            goodpow = np.where(np.abs(self.periodogramP-self.peaksP[i]) ==
                               np.min(np.abs(self.periodogramP-
                                             self.peaksP[i])))[0]
            plt.axhline(self.periodogrampow[goodpow], color='k', ls='-', lw=.8)
            plt.text(1e2+i*1e2, self.periodogrampow[goodpow]+6e-2,
                     'FAP(P = %.3f) = %.3f'%(self.peaksP[i],
                                             self.peaksfaps[i]), fontsize=11)
        plt.xscale('log')
        plt.xlim((1,2e3))
        plt.xlabel('Period (days)')
        plt.ylabel('Power (arbitrary units)')
        plt.subplots_adjust(bottom=.12)
        if label:
            plt.savefig('plots/periodogram_'+self.outsuffix+'.png')
            plt.savefig('plots/periodogram_'+self.outsuffix+'.eps', format='eps')
        if pltt:
            plt.show()
        plt.close('all')


    def plot_RVpert(self, label=False, pltt=False):
        '''Plot the estimated RV perturbation due to starspots (convBS + 
        flux suppression).'''
        plt.plot(self.modelbjd, self.dRV_rotmean, 'b-', label='Flux')
        plt.plot(self.modelbjd, self.dRV_cmean, 'g-', label='Conv. blueshift')
        plt.plot(self.modelbjd, self.dRVtotmean, 'k-', lw=2.5, label='Total')
        plt.fill_between(self.modelbjd, self.dRVtot3sig, self.dRVtotm3sig, 
                         color='k', alpha=.4)
        plt.xlabel('BJD')
        plt.ylabel('$\Delta$RV (m/s)')
        plt.minorticks_on()
        plt.legend(loc='upper right')
        if label:
            plt.savefig('plots/dRV_'+self.outsuffix+'.png')
            plt.savefig('plots/dRV_'+self.outsuffix+'.eps', format='eps')
        if pltt:
            plt.show()
        plt.close('all')


    def plot_GPphot_dRV(self, label=False, pltt=False):
        '''Plot the MEarth photometry, GP model, and corresponding RV spot 
        model on the same time axis.'''
        fig = plt.figure(figsize=(11,6))
        # Plot photometry
        t0 = min(self.bjdtrim)
        ax1 = fig.add_subplot(211)
        ax1.plot(self.bjdtrim-t0, self.magtrim, 'k.', alpha=.1)
        ax1.errorbar(self.bjdtrimbin-t0, self.magtrimbin, self.emagtrimbin, 
                     fmt='bo', capsize=0)
        # Plot GP model
        ax1.plot(self.modelbjd-t0, self.model, 'g-', lw=2)
        ax1.fill_between(self.modelbjd-t0, self.model+3*self.modelerr, 
                         self.model-3*self.modelerr, color='g', alpha=.5)
        # Plot GP prediction
        ax1.plot(self.predbjd-t0, self.predmodel, 'k-', lw=2)
        ax1.fill_between(self.predbjd-t0, self.predmodel+3*self.predmodelerr,
                         self.predmodel-3*self.predmodelerr, color='k', 
                         alpha=.5)
        ax1.set_ylim((.025, -.025))
        ax1.set_xlim((-10, max(self.predbjd)-t0))
        ax1.set_ylabel('Differential\nMagnitude')
        ax1.minorticks_on()
        # Plot RV spot model
        ax2 = fig.add_subplot(212)
        ax2.plot(self.modelbjd-t0, self.dRV_rotmean, 'k--', label='Flux')
        ax2.plot(self.modelbjd-t0, self.dRV_cmean, 'k-.', 
                 label='Conv. blueshift')
        ax2.plot(self.modelbjd-t0, self.dRV_totmean, 'g-', lw=2, label='Total')
        ax2.fill_between(self.modelbjd-t0, self.dRV_totmean+3*self.dRV_totsig,
                         self.dRV_totmean-3*self.dRV_totsig, color='g', alpha=.5)
        # Plot RV spot prediction
        ax2.plot(self.predbjd-t0, self.dRV_totpredmean, 'k-')
        ax2.fill_between(self.predbjd-t0, self.dRV_totpredmean+3*self.dRV_totpredsig,
                         self.dRV_totpredmean-3*self.dRV_totpredsig, color='g', 
                         alpha=.5)
        ax2.set_xlim((-10, max(self.predbjd)-t0))
        ax2.set_xlabel('BJD - %.5f'%t0)
        ax2.set_ylabel('$\Delta RV$  (m/s)')
        ax2.minorticks_on()
        ax2.legend(loc='upper right')
        ax2.set_yticks(np.round(np.arange(-.6, .5, .2), 1))
        ax2.set_yticklabels(np.round(np.arange(-.6, .5, .2), 1))

        # Highlight peaks for visual clarity
        good=np.array([i for i in range(1, self.model.size-1)
                       if self.model[i-1] < self.model[i]
                       and self.model[i+1] < self.model[i]])
        good=np.append(good, np.array([i for i in range(1, self.model.size-1)
                                       if self.model[i-1] > self.model[i]
                                       and self.model[i+1] > self.model[i]]))
        for i in good:
            ax1.axvline(self.modelbjd[i]-t0, color='k', ls='--', lw=.8)
            ax2.axvline(self.modelbjd[i]-t0, color='k', ls='--', lw=.8)

        plt.subplots_adjust(hspace=0)
        if label:
            plt.savefig('plots/photanddRV_%s.png'%self.outsuffix)
        if pltt:
            plt.show()
        plt.close('all')


    def plot_GPsummary(self, label=False, pltt=False):
        '''Make a corner plot of the GP parameter posteriors and the evolution 
        of the lnprobability = lnlikelihood + lnprior'''
        plt.figure('lnlike')
        plt.plot(self.gplnprobs, 'ko-')
        nwalkers, burnin, nsteps = self.gpvals
        plt.axvline(burnin, color='k', ls='--')
        plt.xlabel('Step Number')
        plt.ylabel('lnlikelihood')
        if label:
            plt.savefig('plots/gplnprob_'+self.outsuffix+'.png')
            plt.savefig('plots/gplnprob_'+self.outsuffix+'.eps', format='eps')      

        # Next plot
        corner.corner(self.gpsamples, bins=40)
        if label:
            plt.savefig('plots/gptri_'+self.outsuffix+'.png')
            plt.savefig('plots/gptri_'+self.outsuffix+'.eps', format='eps')
        if pltt:
            plt.show()
        plt.close('all')

    
    def plot_GPmodel(self, label=False, pltt=False):
        '''Plot the lightcurve and the best-fit (most likely) GP model.'''
        # Plot data and model
        plt.close('all')
        plt.figure(figsize=(12,5))
    	t0 = min(self.bjd)
        plt.plot(self.bjdtrim-t0, self.magtrim, 'k.', alpha=.1)
        plt.errorbar(self.bjdtrimbin-t0, self.magtrimbin, self.emagtrimbin, 
                     fmt='bo', capsize=0)
        plt.plot(self.modelbjd-t0, self.model, 'g-', lw=2, label='QP GP')
        plt.plot(self.modelbjd-t0, self.model-3*self.modelerr, 'g-', lw=.7)
        plt.plot(self.modelbjd-t0, self.model+3*self.modelerr, 'g-', lw=.7)
        plt.fill_between(self.modelbjd-t0, self.model-3*self.modelerr, 
                         self.model+3*self.modelerr, color='g', alpha=.5)

        # Plot prediction
        plt.plot(self.predbjd-t0, self.predmodel, 'k-', lw=2, 
                 label='GP prediction')
        plt.fill_between(self.predbjd-t0, self.predmodel-3*self.predmodelerr, 
                         self.predmodel+3*self.predmodelerr, color='k',
                         alpha=.5)
        
        plt.gca().invert_yaxis()
        plt.xlabel('BJD - %.5f'%t0)
        plt.ylabel('Differential Magnitude')
        #plt.legend(loc='lower center')
        plt.xlim((-10,max(self.predbjd)-t0))
	plt.ylim((.025,-.025))
	plt.minorticks_on()
	plt.subplots_adjust(bottom=.15, top=.93)
        if label:
            plt.savefig('plots/gpmodel_'+self.outsuffix+'.png')
            plt.savefig('plots/gpmodel_'+self.outsuffix+'.eps', format='eps')
        if pltt:
            plt.show()
        plt.close('all')

        
    ##########################################################################
    ## Save important information
    ##########################################################################
    def saveGP4RV(self):
        '''Save the GP samples to be used by other GPs including one to 
        model the exising HARPS observations.'''
        np.savetxt('../HARPS/data/QPGPsamples_'+self.outsuffix, self.gpsamples,
                   delimiter='\t', fmt='%.6e', header='lna, lnl, lnG, lnP')


    def pickleobject(self):
        '''Save the complete object to a binary file.'''
        fObj = open('pickles/GJ1132_'+self.outsuffix, 'wb')
        pickle.dump(self, fObj)
        fObj.close()


def loadpickle(fname):
    fObj = open(fname, 'rb')
    self = pickle.load(fObj)
    fObj.close()
    return self


##########################################################################
## Main program
##########################################################################
if __name__ == '__main__':
    #data = MEarthphotometry(outsuffix='testdummy_gponly', GPonly=1) 

    # Try and find periodicities via LS-periodogram
    #data.compute_periodogram()
    #data.plot_periodogram(label=1, pltt=1)
    #data.runqpgp(nsteps=1000, burnin=500, nwalkers=36)
    #data.get_model()
    #data.estimate_RVpert2()
    #data.plot_GPsummary(label=1, pltt=1)
    #data.plot_GPmodel(label=1, pltt=1)
    #data.pickleobject()    
    #data/saveGP4RV()
    
    f=open('pickles/GJ1132_testdummy_gponly', 'rb')
    data=pickle.load(f)
    f.close()

    data.estimate_RVpert3()
    data.plot_GPphot_dRV(label=1, pltt=1)
    data.pickleobject()
