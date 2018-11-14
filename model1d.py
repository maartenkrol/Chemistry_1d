import matplotlib
matplotlib.rcParams.update({'font.size': 12})
try:
    from ipywidgets import *
except:
    from IPython.html.widgets import *
import subprocess,glob,os
from binas import *
from numpy import *
from datetime import datetime
from datetime import timedelta
from copy import *
from pylab import *
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import column, row
from bokeh.plotting import figure



class model_1d:

    def __init__(self):
        self.avo3  = 6.022e26 # molecules/kmol
        self.dz = 1000.0  # m
        self.ch4 = []
        self.co = []
        self.rh = []
        self.nox = []
        self.o3 = []
        self.oh = []
        self.ho2 = []
        self.no = []
        self.no2 = []
        self.scenario = []
        self.emissions = []
        self.isim = 0
        self.color = ['black','blue','red','green','pink','brown','coral','darkcyan','drakgrey']
        self.eco = FloatSlider(description='E CO (Tg CO/year)', min=0., max=5000., step=10.0, value = 1250.0)
        self.eno = FloatSlider(description='E NO (Tg N/year)', min=0., max=200., step=1.0, value=39.0)
        self.ech4 = FloatSlider(description='E CH4 (Tg CH4/year)', min=0., max=5000., step=10.0, value = 570.0)
        self.enmvoc = FloatSlider(description='E NMVOC (Tg CO eq./year)', min=0., max=5000., step=10.0, value = 1000.)
        self.enmvoc.width=3000
        options = ['CH4','CO','O3','NOx','NMVOC','OH','HO2','NO','NO2']
        xtext = 'Standard'
# use Text from widgets:
        try:
            from ipywidgets import Text
        except:
            from IPython.html.widgets import Text
        self.wsim = Text(description='Simulation',value = xtext)
        self.wsim.width = 1000
        
        # setup plot...
        self.pch4 = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pco = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.po3 = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pnmvoc = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pnox = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.poh = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pho2 = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pno = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )
        self.pno2 = figure(title="Atmospheric Profile", plot_height=300, plot_width=300, y_range=(0,15) )


        z, nd, ch4, co, rh, nox, o3, oh, ho2, no, no2 = self.model1d(1250.0, 39.0, 570.0, 1000.)
        self.pch4.line(ch4*1e9,z,line_width = 2, color = self.color[self.isim])
        self.pch4.yaxis.axis_label = 'z (km)'
        self.pch4.xaxis.axis_label = 'CH4 (ppb)'
        self.pco.line(co*1e9,z,line_width = 2, color = self.color[self.isim])
        self.pco.yaxis.axis_label = 'z (km)'
        self.pco.xaxis.axis_label = 'CO (ppb)'
        self.pnmvoc.line(rh*1e9,z,line_width = 2,color = self.color[self.isim],legend = xtext)
        self.pnmvoc.yaxis.axis_label = 'z (km)'
        self.pnmvoc.xaxis.axis_label = 'NMVOC (ppb)'
        self.pnmvoc.legend.location = "top_right"
        self.po3.line(o3*1e9,z,line_width = 2,color = self.color[self.isim])
        self.po3.yaxis.axis_label = 'z (km)'
        self.po3.xaxis.axis_label = 'O3 (ppb)'
        self.pnox.line(nox*1e9,z,line_width = 2,color = self.color[self.isim])
        self.pnox.yaxis.axis_label = 'z (km)'
        self.pnox.xaxis.axis_label = 'NOx (ppb)'
        self.poh.line(oh*nd*1e-6,z,line_width = 2,color = self.color[self.isim])
        self.poh.yaxis.axis_label = 'z (km)'
        self.poh.xaxis.axis_label = 'OH (10^6 molec/cm3)'
        self.pho2.line(ho2*1e12,z,line_width = 2,color = self.color[self.isim])
        self.pho2.yaxis.axis_label = 'z (km)'
        self.pho2.xaxis.axis_label = 'HO2 (ppt)'
        self.pno.line(no*1e9,z,line_width = 2,color = self.color[self.isim])
        self.pno.yaxis.axis_label = 'z (km)'
        self.pno.xaxis.axis_label = 'NO (ppb)'
        self.pno2.line(no2*1e9,z,line_width = 2,color = self.color[self.isim])
        self.pno2.yaxis.axis_label = 'z (km)'
        self.pno2.xaxis.axis_label = 'NO2 (ppb)'

        show(column(row(self.pch4,self.pco,self.po3),
                    row(self.pnox,self.pnmvoc,self.poh),
                    row(self.pho2,self.pno,self.pno2)), notebook_handle=True)

        interact_manual(self.model1d_widget,emission_co=self.eco, emission_no=self.eno,
                                       emission_ch4=self.ech4, emission_nmvoc=self.enmvoc, 
                                       Simulation=self.wsim)








    def atmosphere(self):
        from binas import xmair
        from numpy.linalg import inv
        f = open('atmmod.afglmw.100','r')
        lines = f.readlines()
        f.close()
        vals = []
        for line in lines[2:]:
           vals.append([float(val) for val in line.split()[0:]])
        vals.reverse()
        n = 50
        vals = array(vals)
        zi = vals[:n+1,0]
        z  = 0.5*(vals[:n,0] + vals[1:n+1,0])
        temp = 0.5*(vals[1:n+1,2]+vals[:n,2])
        pres = 0.5*(vals[1:n+1,1]+vals[:n,1])*1e2  #Pa
        nd = zeros((n))
        nd = 0.5*(vals[:n,3]+vals[1:n+1,3])  # #/cm3
        ndh = vals[:n+1,3]
        dens = nd*xmair/self.avo3/1e-6  # kg/m3
        h2o = 0.5*(vals[:n,6]+vals[1:n+1,6])  # #/cm3
        #print nd
        #print ndh
        kz = zeros((n+1))
        kz[0] = 0.0
        kz[1] = 10.0   # m2/s     Boundary layer
        kz[2] = 5.0   # m2/s      Upper boundary layer
        kz[3:12] = 1.0  # m2/s    Free atmosphere
        kz[12:-1] = 0.1   # m2/s  Stratosphere
# setup transport matrix     (d2C/dz2):   dC(i+1/2)/dz = C(i+1) - C(i)/dz
# dC(i-1/2)/dz = C(i) - C(i-1) / dz : d2C(i)/dz = dC(i+1/2)/dz - dC(i-1/2)/dz / dz
# =   (C(i-1) - 2*C(i) + C(i+i))/(dz)^2

        return z, zi, temp, pres, dens, nd, ndh, h2o

# simple arrhenius...
    def zfarr(self,rx1,er,ztrec):
        #------------------------------------------------------------------
        #     
        #****  ZFARR calculation of Arrhenius expression for rate constants
        #
        #------------------------------------------------------------------
        #
        zfarr=rx1*exp(er*ztrec)
        return zfarr
# calculate rate constrants (pressure / temp dependent)

    def rates(self,temp,nd,h2o,pres):
        ztrec = 1./temp
        knoo3 = self.zfarr(3.e-12,-1500.,ztrec)
        kho2no = self.zfarr(3.5e-12,250.,ztrec)
        kodm = 0.2095*self.zfarr(3.3e-11,55.,ztrec)+ 0.7808*self.zfarr(2.15e-11,110.,ztrec)
        kh2ood = self.zfarr(1.63e-10,60.,ztrec)
        tojeff = h2o*kh2ood/(kodm*nd + kh2ood*h2o)
        kcooh = 1.5e-13 + 9e-14*pres/101325.
        rfactor = 10.0
        krhoh = kcooh*rfactor
        kch4oh = self.zfarr(2.45e-12,-1775,ztrec)
        ko3ho2 = self.zfarr(1.0e-14,-490.,ztrec)
        ko3oh = self.zfarr(1.7e-12,-940.,ztrec)
        kho2oh = self.zfarr(4.8e-11,250.,ztrec)
        k1 =self.zfarr(3.5e-13,430.,ztrec)
        k2 =self.zfarr(1.7e-33,1000.,ztrec)
        k3 =self.zfarr(1.4e-21,2200.,ztrec)
        kho2ho2 = (k1 + k2*nd)*(1. + k3*h2o)
        kro2ho2 =self.zfarr(4.1e-13,750.,ztrec) 
        kro2no  =self.zfarr(2.8e-12,300.,ztrec)       
        kro2ro2 =self.zfarr(9.5e-14,390.,ztrec)      
        rx1 = nd*1.8e-30*(300./temp)**3.0
        rx2 = 2.8e-11
        rx3 = 0.6
        kno2oh = rx1/(1.+rx1/rx2)*rx3**(1./(1.+log10(rx1/rx2)**2))
        k2 = 1e-33
        k4 = 8e-12*exp(-2060/temp)
        return knoo3,kho2no,tojeff,kcooh,krhoh,kch4oh,ko3ho2,ko3oh,kho2oh,kho2ho2, \
               kro2ho2,kro2no,kro2ro2,kno2oh,k2,k4
# main program
    def model1d(self,emission_co,emission_no,emission_ch4,emission_nmvoc):

        z, zi, temp, pres, dens, nd, ndh, h2o = self.atmosphere()
        n = shape(z)[0]
        ma = zeros((n))   # airmass above level
        ma[n-1] = 0.5*nd[n-1]*self.dz   # molecules/cm2
        for i in range(n-2,-1,-1):
           ma[i] = ma[i+1] + 0.5*(nd[i+1]+nd[i])*self.dz
        so2 = 1.5e-21   # cm2/#
        so3uvc = 7e-17
        so3uvb = 4e-17
        sno2 = 1e-18 
        ssao3 = 1.0e-23
        ssano2 = 6.0e-24
        jo2 = 8e-10*exp(-ma*so2)
        
        knoo3,kho2no,tojeff,kcooh,krhoh,kch4oh,ko3ho2,ko3oh,kho2oh,kho2ho2, \
           kro2ho2,kro2no,kro2ro2,kno2oh,k2,k4 = self.rates(temp,nd,h2o,pres)

        o3 = 10e-6*nd   # initial o3 in #/cm3
        o  = zeros((n))
        o2 = 0.21*nd
        iter = 0
# calculate steady state ozone (Jacob) layer:
        while iter < 20:
           iter += 1
           mo3 = zeros((n))   # O3 molecules/cm2 above level
           mo3[n-1] = 0.5*o3[n-1]*self.dz   # molecules/cm2
           for i in range(n-2,-1,-1):
              mo3[i] = mo3[i+1] + 0.5*(o3[i+1]+o3[i])*self.dz
           jo3 = 1.6e-2*exp(-mo3*so3uvc) +8e-4*exp(-so3uvb*mo3-ssao3*ma)  # ozone photolysis (depends on O3/O2)
           o3 = sqrt(jo2*k2/(jo3*k4))*0.21*nd**1.5

 #       print ('Chapman O3', sum(o3*1e6)*1000./6.022e23*0.02224*1e5,' Du')
        mo3 = zeros((n))   # O3 molecules/cm2 above level
        mo3[n-1] = 0.5*o3[n-1]*self.dz   # molecules/cm2
        for i in range(n-2,-1,-1):
           mo3[i] = mo3[i+1] + 0.5*(o3[i+1]+o3[i])*self.dz
        jo3 = 1.6e-2*exp(-mo3*so3uvc) +8e-4*exp(-so3uvb*mo3-ssao3*ma)  # ozone photolysis (depends on O3/O2)
        jno2 = 1e-2*exp(-mo3*sno2-ssano2*ma)
        tt = 0
        #plot(o3/nd,z)

        n = 15
        nd = nd[:n]  # lowest n km
        ndh = ndh[:n+2]
        # account for tropics...
        temp = temp[:n] + 10.0
        h2o = h2o[:n]
        pres = pres[:n]
        jo3 = jo3[:n]
        jo2 = jo2[:n]
        # account for the fact that we simulate the tropics...!
        jno2 = jno2[:n]*2.0
        z = z[:n]

        knoo3,kho2no,tojeff,kcooh,krhoh,kch4oh,ko3ho2,ko3oh,kho2oh,kho2ho2, \
           kro2ho2,kro2no,kro2ro2,kno2oh,k2,k4 = self.rates(temp,nd,h2o,pres)

        jo3e = jo3*tojeff
        # enhance: we are in the tropics
        jo3e = jo3e*2.0
# to mixing ratio units:
        knoo3 *= nd
        kho2no *= nd
        kcooh *= nd 
        krhoh *= nd
        kch4oh *= nd
        ko3ho2 *= nd
        ko3oh *= nd
        kho2oh *= nd
        kho2ho2 *= nd
        kro2ho2 *= nd
        kro2no *= nd
        kro2ro2 *= nd
        kno2oh *= nd


# calculate in mr units:
        no2 = zeros((n))
        no  = zeros((n)) 
        nox  = zeros((n)) 
        co  = zeros((n))
        rh  = zeros((n))
        ch4 = zeros((n))
        o3top = 500.0e-9
        o3  = o3[:n]/nd
        no[:] = 1e-12
        no2[:] = 1e-12
        nox = no + no2
        co[:] = 40e-9
        rh[:] = 0.0
        ch4_fixed = 1700.0  # ppb
        ch4[:]  = ch4_fixed*1e-9
#deposition...
        vdo3 = zeros((n))
        vdo3[0] = 2e-3   # m/s
        vdo3 /= self.dz   # 1/s
        vdno2 = zeros((n))
        vdno2[0] = 2e-3  # m/s
        vdno2[1:] = 1e-4  # m/s wet removal nox to aerosols...
        vdno2 /= self.dz    # cm/s
        #f = open('input','r')
        #input = f.readlines()
        #f.close()
        #en_t = float(input[0].split()[0])
        en_t = emission_no
        enl_t   = 5.0    # Tg/yr, globally lightning
        ech4_t = emission_ch4
        eco_t = emission_co
        erh_t = emission_nmvoc
        #ech4_t = float(input[1].split()[0])
        #eco_t = float(input[2].split()[0])
        #erh_t = float(input[3].split()[0])
#    ech4_t = 500.0   # Tg/yr
#    eco_t  = 800.0  # Tg/yr, globally
#    erh_t = 400.0      # Tg/yr CO equiv., globally
        area = 4e4*pi*radius**2   # cm2
        conv = 1e12*6.022e23/(area*3600.*24.*365.)   # from Tg/yr --> #/cm2/s (if divided by molmass in g)
        eno  = zeros((n))
        eno[0] = en_t*conv/xmn            # surface nox
        prof = nd[0:12]/sum(nd[0:12])
        eno[0:12] += (enl_t*prof)*conv/xmn  # lightning nox
        eco  = zeros((n))
        eco[0] = eco_t*conv/(xmc + xmo)
        erh  = zeros((n))
        erh[0] = erh_t*conv/(xmc + xmo)

        ech4  = zeros((n))
        ech4[0] = ech4_t*conv/(xmc + 4*xmh)

        eno /= (nd*100*self.dz)
        eco /= (nd*100*self.dz)
        erh /= (nd*100*self.dz)
        ech4 /= (nd*100*self.dz)   # mixing ratio / s
# now find the solution by setting dC/dt = 0
        kz = zeros((n+1))
        kz[0] = 0.0
        kz[1] = 10.0   # m2/s
        kz[2] = 5.0   # m2/s
        kz[3:12] = 1.0  # m2/s
        kz[12:-1] = 0.1   # m2/s
        kztop = 0.1/(self.dz*self.dz)
        kz /= (self.dz*self.dz)    #  1/s
#      C in mixing ratio, t in seconds
#      dC/dt = P - L.C + d/dz K dC/dz  = 0.0, later calculate directly full solution: now keep it simple (and slow..):

        oh = zeros((n))
        oh[:] = 1e-12
        ho2 = zeros((n))
        ro2 = zeros((n))

        con = zeros((n))
        ch4n = zeros((n))
        noxn = zeros((n))
        rhn = zeros((n))
        o3n = zeros((n))
# first get the non-transported species in SS
        ratio = 0.1  # dampening factor...
#setup plot:
        iter = 0
        convergence = 1.0
# species  CH4(0), CO(1), RH(2), NO(3), NO2(4), O3(5), OH(6), HO2(7), RO2(8)
# set-up system of equations dC/dt =  d/dz K dC/dz + P(C) - L.C + E = 0
#  dch4[0]/dt = E - oh[0]*kch4oh[0]*ch4[0] + ech4[0]
#       ch4n[0] = (pch4[0] + ch4[1]*kz[1]*ndh[1]/nd[0])/(lch4[0] + kz[1]*ndh[1]/nd[0])

        while (iter < 20000 and convergence > 1e-12):
           iter += 1
           #print(iter,convergence,ch4[0]*1e6)
           poh = (jo3e*o3 + kho2no*no*ho2 + ko3ho2*ho2*o3 + kro2no*no*ro2)
           loh = (kcooh*co + krhoh*rh + kch4oh*ch4 + kno2oh*no2 + ko3oh*o3 + kho2oh*ho2)
           oh = poh/loh   # first guess
           pho2 = (kcooh*co*oh + ko3oh*o3*oh)
           lho2 = (kho2no*no + 2*kho2ho2*ho2 + ko3ho2*o3 + kro2ho2*ro2 + kho2oh*oh)
           ho2 = pho2/lho2  # first guess
           pro2 = (kch4oh*ch4*oh + krhoh*oh*rh)
           lro2 = (2*kro2ro2*ro2 + kro2ho2*ho2 + kro2no*no)
           ro2 = pro2/lro2   #first guess
           po3 = (kho2no*no*ho2 + kro2no*ro2*no)
           lo3 = (jo3e + ko3ho2*ho2 + ko3oh*oh + vdo3)
           pnox = (eno)
           lnox = (vdno2 + kno2oh*oh)*no2/nox   # only apply at the fraction NO2/NOX
           r_nono2 = (jno2)/(knoo3*o3 + kho2no*ho2 + kro2no*ro2)
# no + no2 = nox; no/no2 = r
# no2 ( 1 + r ) = nox
           pco = (eco + 0.9*kch4oh*ch4*oh + 0.5*krhoh*oh*rh)
           lco = (kcooh*oh)
           prh = (erh)
           lrh = (krhoh*oh)
           pch4 = (ech4)
           lch4 = (kch4oh*oh)
           mat = zeros((n,n))
           mat[0,0] = -lch4[0] - kz[1]*ndh[1]/nd[0]
           mat[0,1] = kz[1]*ndh[1]/nd[0]
           for l in range(1,n-1):
               mat[l,l] = -lch4[l] - kz[l]*ndh[l]/nd[l] - kz[l+1]*ndh[l+1]/nd[l]
               mat[l,l+1] = kz[l+1]*ndh[l+1]/nd[l]  
               mat[l,l-1] = kz[l]*ndh[l]/nd[l]  
           mat[n-1,n-1] = -lch4[n-1] - kz[n-1]*ndh[n-1]/nd[n-1]
           mat[n-1,n-2] =  kz[n-1]*ndh[n-1]/nd[n-1] 
           ch4n = dot(inv(mat),-pch4)   # solution dch4/dt = 0.0

           con[0] = (pco[0] + co[1]*kz[1]*ndh[1]/nd[0])/(lco[0] + kz[1]*ndh[1]/nd[0])
           rhn[0] = (prh[0] + rh[1]*kz[1]*ndh[1]/nd[0])/(lrh[0] + kz[1]*ndh[1]/nd[0])
           noxn[0] = (pnox[0] + nox[1]*kz[1]*ndh[1]/nd[0])/(lnox[0] + kz[1]*ndh[1]/nd[0])
           o3n[0] = (po3[0] + o3[1]*kz[1]*ndh[1]/nd[0])/(lo3[0] + kz[1]*ndh[1]/nd[0])
           for l in range(1,n-1):
              con[l] = (pco[l] + co[l-1]*kz[l]*ndh[l]/nd[l] + co[l+1]*kz[l+1]*ndh[l+1]/nd[l])/ \
                    (lco[l] + kz[l]*ndh[l]/nd[l] + kz[l+1]*ndh[l+1]/nd[l])
              rhn[l] = (prh[l] + rh[l-1]*kz[l]*ndh[l]/nd[l] + rh[l+1]*kz[l+1]*ndh[l+1]/nd[l])/ \
                    (lrh[l] + kz[l]*ndh[l]/nd[l] + kz[l+1]*ndh[l+1]/nd[l])
              noxn[l] = (pnox[l] + nox[l-1]*kz[l]*ndh[l]/nd[l] + nox[l+1]*kz[l+1]*ndh[l+1]/nd[l])/ \
                    (lnox[l] + kz[l]*ndh[l]/nd[l] + kz[l+1]*ndh[l+1]/nd[l])
              o3n[l] = (po3[l] + o3[l-1]*kz[l]*ndh[l]/nd[l] + o3[l+1]*kz[l+1]*ndh[l+1]/nd[l])/ \
                    (lo3[l] + kz[l]*ndh[l]/nd[l] + kz[l+1]*ndh[l+1]/nd[l])
           con[n-1] = (pco[n-1] + co[n-2]*kz[n-1]*ndh[n-1]/nd[n-1])/(lco[n-1] + kz[n-1]*ndh[n-1]/nd[n-1])
           rhn[n-1] = (prh[n-1] + rh[n-2]*kz[n-1]*ndh[n-1]/nd[n-1])/(lrh[n-1] + kz[n-1]*ndh[n-1]/nd[n-1])
           noxn[n-1] = (pnox[n-1] + nox[n-2]*kz[n-1]*ndh[n-1]/nd[n-1] )/(lnox[n-1] + kz[n-1]*ndh[n-1]/nd[n-1])
           #o3n[n-1] = (po3[n-1] + o3[n-2]*kz[n-1]*ndh[n-1]/nd[n-1] ) /(lo3[n-1] + kz[n-1]*ndh[n-1]/nd[n-1])
           #noxn[n-1] = 1e-9
           o3n[n-1] = 100.e-9

           convergence = abs(ch4n-ch4).sum()
           o3 = ratio*o3 + (1-ratio)*o3n
           co = ratio*co + (1-ratio)*con
           nox = ratio*nox + (1-ratio)*noxn
           rh = ratio*rh + (1-ratio)*rhn
           no2 = nox/(1. + r_nono2)
           no = nox-no2
           oh = ratio*oh + (1-ratio)*poh/loh
           ho2 = ratio*ho2 + (1-ratio)*pho2/lho2
           ro2 = ratio*ro2 + (1-ratio)*pro2/lro2
           ch4 = ch4n

        print ('Reached steady state in %i8 iterations with a surface methane mixing ratio of %8.1f ppb'%(iter,ch4[0]*1e9))
        return z, nd, ch4, co, rh, nox, o3, oh, ho2, no, no2
        
    def model1d_widget(self,emission_co=1250.,emission_no=39., emission_ch4=570., emission_nmvoc=1000., Simulation = Text()):


        z, nd, ch4, co, rh, nox, o3, oh, ho2, no, no2 =  \
            self.model1d(emission_co,emission_no,emission_ch4, emission_nmvoc)

# test if these emissions have already been processed?
        current_emissions = array([emission_co,emission_no,emission_ch4,emission_nmvoc])
        store = True
        for i,emis in enumerate(self.emissions):
            if all(current_emissions == emis):
                store = False
                #  it could be a renaming action:
                self.scenario[i] = Simulation
        if store:
            self.ch4.append(ch4)
            self.co.append(co)
            self.nox.append(nox)
            self.rh.append(rh)
            self.o3.append(o3)
            self.oh.append(oh)
            self.ho2.append(ho2)
            self.no.append(no)
            self.no2.append(no2)
            self.scenario.append(Simulation)
            self.emissions.append(current_emissions)
            self.isim += 1
        self.pch4.line(ch4*1e9,z,line_width = 2, color = self.color[self.isim])
        self.pco.line(co*1e9,z,line_width = 2, color = self.color[self.isim])
        self.pnmvoc.line(rh*1e9,z,line_width = 2,color = self.color[self.isim],legend = Simulation)
        self.po3.line(o3*1e9,z,line_width = 2,color = self.color[self.isim])
        self.pnox.line(nox*1e9,z,line_width = 2,color = self.color[self.isim])
        self.poh.line(oh*nd*1e-6,z,line_width = 2,color = self.color[self.isim])
        self.pho2.line(ho2*1e12,z,line_width = 2,color = self.color[self.isim])
        self.pno.line(no*1e9,z,line_width = 2,color = self.color[self.isim])
        self.pno2.line(no2*1e9,z,line_width = 2,color = self.color[self.isim])
        push_notebook()

