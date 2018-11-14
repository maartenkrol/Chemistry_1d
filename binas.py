from numpy import pi

radius  = 6.371e6	# the earth radius in meters
deg2rad = pi/180.   # convert degrees to radians
g       = 9.80665   # accelaration of gravity

xmair=28.9        # mass of air
xmh=1.0079          # mass of H
xmn=14.0067         # mass of N
xmc=12.01115        # mass of C
xms=32.064          # mass of N
xmo=15.9994         # mass of O
xmcl=35.453         # mass of Cl
xmf=18.9984032      # mass of F
xmh2o=xmh*2+xmo     # mass of H2O
xmco2=xmc+2*xmo     # mass of CO2
xmrn222=222.        # mass of 222Rn
xmpb210=210.        # mass of 210Pb
xmsf6= xms+6*xmf    # mass of SF6

kgCs_to_PgCyr=86400.*365./1.e12   # convert kgC/s to PgC/yr
kgCs_to_mmols=1.e9/xmc            # convert kgC/s to umol/s
