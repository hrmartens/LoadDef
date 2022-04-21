# Function to plot displacement results for several disk loads
# H.R. Martens 2021-2022

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# Import Python Modules
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb
import matplotlib.cm as cm
import numpy as np
import os
import sys
from CONVGF.utility import read_convolution_file

# Convolution files
# UNCOMMENT SEQUENCE OF LINES:
prefix = ("LandAndOceans_")
suffix = ("km-NoTaperce_convgf_disk")
disk_rad = 1
label1 = '1 km'
mod1 = (prefix + str(disk_rad) + suffix)
disk_rad = 2
label2 = '2 km'
mod2 = (prefix + str(disk_rad) + suffix)
disk_rad = 5
label3 = '5 km'
mod3 = (prefix + str(disk_rad) + suffix)
disk_rad = 10
label4 = '10 km'
mod4 = (prefix + str(disk_rad) + suffix)
disk_rad = 20
label5 = '20 km'
mod5 = (prefix + str(disk_rad) + suffix)
disk_rad = 30
label6 = '30 km'
mod6 = (prefix + str(disk_rad) + suffix)
disk_rad = 50
label7 = '50 km'
mod7 = (prefix + str(disk_rad) + suffix)
disk_rad = 100
label8 = '100 km'
mod8 = (prefix + str(disk_rad) + suffix)
disk_rad = 200
label9 = '200 km'
mod9 = (prefix + str(disk_rad) + suffix)
# SAME FOR ALL:
cfile1 = ("../pmes/output/cn_" + mod1 + ".txt")
cfile2 = ("../pmes/output/cn_" + mod2 + ".txt")
cfile3 = ("../pmes/output/cn_" + mod3 + ".txt")
cfile4 = ("../pmes/output/cn_" + mod4 + ".txt")
cfile5 = ("../pmes/output/cn_" + mod5 + ".txt")
cfile6 = ("../pmes/output/cn_" + mod6 + ".txt")
cfile7 = ("../pmes/output/cn_" + mod7 + ".txt")
cfile8 = ("../pmes/output/cn_" + mod8 + ".txt")
cfile9 = ("../pmes/output/cn_" + mod9 + ".txt")
c1 = 'black'
c2 = 'blue'
c3 = 'deeppink'
c4 = 'gold'
c5 = 'green'
c6 = 'firebrick'
c7 = 'darkorange'
c8 = 'purple'
c9 = 'navy'
# RANGE:
xmin = 89.0
xmax = 90.0
yminh = 0.0
ymaxh = 10.0
yminu = -30.0
ymaxu = 2.0
# Use Co-Latitude?
colat = True
# Figure names
plot_title = ("Disks | PREM | 1 m Fresh Water")
plot_name = ("./output/all_disks_1m_PREM_" + str(int(90-xmax)) + "_" + str(int(90-xmin)) + ".pdf")
plot_name_NU = ("./output/all_disks_NU_1m_PREM_" + str(int(90-xmax)) + "_" + str(int(90-xmin)) + ".pdf")
 
#### BEGIN CODE

# Read the file
extension1,lat1,lon1,eamp1,epha1,namp1,npha1,vamp1,vpha1 = read_convolution_file.main(cfile1)
extension2,lat2,lon2,eamp2,epha2,namp2,npha2,vamp2,vpha2 = read_convolution_file.main(cfile2)
extension3,lat3,lon3,eamp3,epha3,namp3,npha3,vamp3,vpha3 = read_convolution_file.main(cfile3)
extension4,lat4,lon4,eamp4,epha4,namp4,npha4,vamp4,vpha4 = read_convolution_file.main(cfile4)
extension5,lat5,lon5,eamp5,epha5,namp5,npha5,vamp5,vpha5 = read_convolution_file.main(cfile5)
extension6,lat6,lon6,eamp6,epha6,namp6,npha6,vamp6,vpha6 = read_convolution_file.main(cfile6)
extension7,lat7,lon7,eamp7,epha7,namp7,npha7,vamp7,vpha7 = read_convolution_file.main(cfile7)
extension8,lat8,lon8,eamp8,epha8,namp8,npha8,vamp8,vpha8 = read_convolution_file.main(cfile8)
extension9,lat9,lon9,eamp9,epha9,namp9,npha9,vamp9,vpha9 = read_convolution_file.main(cfile9)

# Combine amp and pha
eamp1 = np.multiply(eamp1,np.cos(epha1*(np.pi/180.)))
namp1 = np.multiply(namp1,np.cos(npha1*(np.pi/180.)))
vamp1 = np.multiply(vamp1,np.cos(vpha1*(np.pi/180.)))
eamp2 = np.multiply(eamp2,np.cos(epha2*(np.pi/180.)))
namp2 = np.multiply(namp2,np.cos(npha2*(np.pi/180.)))
vamp2 = np.multiply(vamp2,np.cos(vpha2*(np.pi/180.)))
eamp3 = np.multiply(eamp3,np.cos(epha3*(np.pi/180.)))
namp3 = np.multiply(namp3,np.cos(npha3*(np.pi/180.)))
vamp3 = np.multiply(vamp3,np.cos(vpha3*(np.pi/180.)))
eamp4 = np.multiply(eamp4,np.cos(epha4*(np.pi/180.)))
namp4 = np.multiply(namp4,np.cos(npha4*(np.pi/180.)))
vamp4 = np.multiply(vamp4,np.cos(vpha4*(np.pi/180.)))
eamp5 = np.multiply(eamp5,np.cos(epha5*(np.pi/180.)))
namp5 = np.multiply(namp5,np.cos(npha5*(np.pi/180.)))
vamp5 = np.multiply(vamp5,np.cos(vpha5*(np.pi/180.)))
eamp6 = np.multiply(eamp6,np.cos(epha6*(np.pi/180.)))
namp6 = np.multiply(namp6,np.cos(npha6*(np.pi/180.)))
vamp6 = np.multiply(vamp6,np.cos(vpha6*(np.pi/180.)))
eamp7 = np.multiply(eamp7,np.cos(epha7*(np.pi/180.)))
namp7 = np.multiply(namp7,np.cos(npha7*(np.pi/180.)))
vamp7 = np.multiply(vamp7,np.cos(vpha7*(np.pi/180.)))
eamp8 = np.multiply(eamp8,np.cos(epha8*(np.pi/180.)))
namp8 = np.multiply(namp8,np.cos(npha8*(np.pi/180.)))
vamp8 = np.multiply(vamp8,np.cos(vpha8*(np.pi/180.)))
eamp9 = np.multiply(eamp9,np.cos(epha9*(np.pi/180.)))
namp9 = np.multiply(namp9,np.cos(npha9*(np.pi/180.)))
vamp9 = np.multiply(vamp9,np.cos(vpha9*(np.pi/180.)))

# Sort by Latitude
lat_idx = np.argsort(lat1)
lat1 = lat1[lat_idx]
lon1 = lon1[lat_idx]
extension1 = extension1[lat_idx]
eamp1 = eamp1[lat_idx]
namp1 = namp1[lat_idx]
vamp1 = vamp1[lat_idx]
lat_idx = np.argsort(lat2)
lat2 = lat2[lat_idx]
lon2 = lon2[lat_idx]
extension2 = extension2[lat_idx]
eamp2 = eamp2[lat_idx]
namp2 = namp2[lat_idx]
vamp2 = vamp2[lat_idx]     
lat_idx = np.argsort(lat3)
lat3 = lat3[lat_idx]
lon3 = lon3[lat_idx]
extension3 = extension3[lat_idx]
eamp3 = eamp3[lat_idx]
namp3 = namp3[lat_idx]
vamp3 = vamp3[lat_idx]
lat_idx = np.argsort(lat4)
lat4 = lat4[lat_idx]
lon4 = lon4[lat_idx]
extension4 = extension4[lat_idx]
eamp4 = eamp4[lat_idx]
namp4 = namp4[lat_idx]
vamp4 = vamp4[lat_idx]
lat_idx = np.argsort(lat5)
lat5 = lat5[lat_idx]
lon5 = lon5[lat_idx]
extension5 = extension5[lat_idx]
eamp5 = eamp5[lat_idx]
namp5 = namp5[lat_idx]
vamp5 = vamp5[lat_idx]
lat_idx = np.argsort(lat6)
lat6 = lat6[lat_idx]
lon6 = lon6[lat_idx]
extension6 = extension6[lat_idx]
eamp6 = eamp6[lat_idx]
namp6 = namp6[lat_idx]
vamp6 = vamp6[lat_idx]
lat_idx = np.argsort(lat7)
lat7 = lat7[lat_idx]
lon7 = lon7[lat_idx]
extension7 = extension7[lat_idx]
eamp7 = eamp7[lat_idx]
namp7 = namp7[lat_idx]
vamp7 = vamp7[lat_idx]
lat_idx = np.argsort(lat8)
lat8 = lat8[lat_idx]
lon8 = lon8[lat_idx]
extension8 = extension8[lat_idx]
eamp8 = eamp8[lat_idx]
namp8 = namp8[lat_idx]
vamp8 = vamp8[lat_idx]
lat_idx = np.argsort(lat9)
lat9 = lat9[lat_idx]
lon9 = lon9[lat_idx]
extension9 = extension9[lat_idx]
eamp9 = eamp9[lat_idx]
namp9 = namp9[lat_idx]
vamp9 = vamp9[lat_idx]

# Colat?
xlab = ("Latitude (deg)")
if (colat == True):
    lat1 = 90-lat1
    lat2 = 90-lat2
    lat3 = 90-lat3
    lat4 = 90-lat4
    lat5 = 90-lat5
    lat6 = 90-lat6
    lat7 = 90-lat7
    lat8 = 90-lat8
    lat9 = 90-lat9
    xmax2 = 90.-xmin
    xmin2 = 90.-xmax
    xmin = xmin2
    xmax = xmax2
    xlab = ("Angular Distance from Load Center (deg)")
    print(xmin)
    print(xmax)
    order = np.argsort(lat1)
    lat1 = lat1[order]
    lon1 = lon1[order]
    eamp1 = eamp1[order]
    namp1 = namp1[order]
    vamp1 = vamp1[order]
    order = np.argsort(lat2)
    lat2 = lat2[order]
    lon2 = lon2[order]
    eamp2 = eamp2[order]
    namp2 = namp2[order]
    vamp2 = vamp2[order]       
    order = np.argsort(lat3)
    lat3 = lat3[order]
    lon3 = lon3[order]
    eamp3 = eamp3[order]
    namp3 = namp3[order]
    vamp3 = vamp3[order]
    order = np.argsort(lat4)
    lat4 = lat4[order]
    lon4 = lon4[order]
    eamp4 = eamp4[order]
    namp4 = namp4[order]
    vamp4 = vamp4[order]
    order = np.argsort(lat5)
    lat5 = lat5[order]
    lon5 = lon5[order]
    eamp5 = eamp5[order]
    namp5 = namp5[order]
    vamp5 = vamp5[order]
    order = np.argsort(lat6)
    lat6 = lat6[order]
    lon6 = lon6[order]
    eamp6 = eamp6[order]
    namp6 = namp6[order]
    vamp6 = vamp6[order]
    order = np.argsort(lat7)
    lat7 = lat7[order]
    lon7 = lon7[order]
    eamp7 = eamp7[order]
    namp7 = namp7[order]
    vamp7 = vamp7[order]
    order = np.argsort(lat8)
    lat8 = lat8[order]
    lon8 = lon8[order]
    eamp8 = eamp8[order]
    namp8 = namp8[order]
    vamp8 = vamp8[order]
    order = np.argsort(lat9)
    lat9 = lat9[order]
    lon9 = lon9[order]
    eamp9 = eamp9[order]
    namp9 = namp9[order]
    vamp9 = vamp9[order]

# Plot
plot_fig = True
fs = 10
if plot_fig:
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(lat1,eamp1,color=c1,label=label1)
    plt.plot(lat2,eamp2,color=c2,label=label2)
    plt.plot(lat3,eamp3,color=c3,label=label3)
    plt.plot(lat4,eamp4,color=c4,label=label4)
    plt.plot(lat5,eamp5,color=c5,label=label5)
    plt.plot(lat6,eamp6,color=c6,label=label6)
    plt.plot(lat7,eamp7,color=c7,label=label7)
    plt.plot(lat8,eamp8,color=c8,label=label8)
    plt.plot(lat9,eamp9,color=c9,label=label9)
    plt.title('East | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.xlim(xmin,xmax)
    plt.ylim(yminh,ymaxh)
    plt.grid(True)
    plt.legend(loc=1,fontsize=4)
    plt.subplot(3,1,2)
    plt.plot(lat1,namp1,color=c1)
    plt.plot(lat2,namp2,color=c2)
    plt.plot(lat3,namp3,color=c3)
    plt.plot(lat4,namp4,color=c4)
    plt.plot(lat5,namp5,color=c5)
    plt.plot(lat6,namp6,color=c6)
    plt.plot(lat7,namp7,color=c7)
    plt.plot(lat8,namp8,color=c8)
    plt.plot(lat9,namp9,color=c9)
    plt.title('North | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.grid(True)
    plt.xlim(xmin,xmax)
    plt.ylim(yminh,ymaxh)
    plt.subplot(3,1,3)
    plt.plot(lat1,vamp1,color=c1)
    plt.plot(lat2,vamp2,color=c2)
    plt.plot(lat3,vamp3,color=c3)
    plt.plot(lat4,vamp4,color=c4)
    plt.plot(lat5,vamp5,color=c5)
    plt.plot(lat6,vamp6,color=c6)
    plt.plot(lat7,vamp7,color=c7)
    plt.plot(lat8,vamp8,color=c8)
    plt.plot(lat9,vamp9,color=c9)
    plt.title('Up | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.xlabel(xlab,fontsize=fs)
    plt.grid(True)
    plt.xlim(xmin,xmax)
    plt.ylim(yminu,ymaxu)
    plt.tight_layout()
    plt.savefig(plot_name,format='pdf')
    plt.show()

    plt.close()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(lat1,namp1,color=c1,linestyle='-',label=label1)
    plt.plot(lat2,namp2,color=c2,linestyle='-',label=label2)
    plt.plot(lat3,namp3,color=c3,linestyle='-',label=label3)
    plt.plot(lat4,namp4,color=c4,linestyle='-',label=label4)
    plt.plot(lat5,namp5,color=c5,linestyle='-',label=label5)
    plt.plot(lat6,namp6,color=c6,linestyle='-',label=label6)
    plt.plot(lat7,namp7,color=c7,linestyle='-',label=label7)
    plt.plot(lat8,namp8,color=c8,linestyle='-',label=label8)
    plt.plot(lat9,namp9,color=c9,linestyle='-',label=label9)
    plt.title(plot_title,fontsize=fs)
    plt.ylabel('Lateral Displacement (mm)',fontsize=fs)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.legend(loc=1,fontsize=6)
    plt.xlim(xmin,xmax)
    plt.ylim(yminh,ymaxh)
    plt.subplot(2,1,2)
    plt.plot(lat1,vamp1,color=c1,linestyle='-')
    plt.plot(lat2,vamp2,color=c2,linestyle='-')
    plt.plot(lat3,vamp3,color=c3,linestyle='-')
    plt.plot(lat4,vamp4,color=c4,linestyle='-')
    plt.plot(lat5,vamp5,color=c5,linestyle='-')
    plt.plot(lat6,vamp6,color=c6,linestyle='-')
    plt.plot(lat7,vamp7,color=c7,linestyle='-')
    plt.plot(lat8,vamp8,color=c8,linestyle='-')
    plt.plot(lat9,vamp9,color=c9,linestyle='-')
    #plt.title(plot_title,fontsize=fs)
    plt.ylabel('Up Displacement (mm)',fontsize=fs)
    plt.xlabel(xlab,fontsize=fs)
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.xlim(xmin,xmax)
    plt.ylim(yminu,ymaxu)
    plt.tight_layout()
    plt.savefig(plot_name_NU,format='pdf')
    plt.show()

