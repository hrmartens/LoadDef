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
prefix = ("LandAndOceans_")
suffix = ("km-NoTaperce_convgf_disk_1m_PREM")
disk_rad = 10
label1 = '10 km'
mod1 = (prefix + str(disk_rad) + suffix)
disk_rad = 25
label2 = '25 km'
mod2 = (prefix + str(disk_rad) + suffix)
mods = [mod1,mod2]
labels = [label1,label2]
colors = ['black','blue']
# colors = ['black','blue','deeppink','gold','green','firebrick','darkorange','purple','navy']
# RANGE:
xmin = 89.5
xmax = 90.0
yminh = 0.0
ymaxh = 1.0
yminu = -5.0
ymaxu = 2.0
# Use Co-Latitude?
colat = True
# Figure names
plot_title = ("Disks | PREM | 1 m Fresh Water")
plot_name_NU = ("./output/all_disks_NU_1m_PREM_" + str(int(90-xmax)) + "_" + str(int(90-xmin)) + ".pdf")

 
#### BEGIN CODE


# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")

# initialize figure
plt.figure()

# loop through models
for ii in range(0,len(mods)):

    # Current model
    cmod = mods[ii]

    # Current file
    cfile = ("../pmes/output/cn_" + cmod + ".txt")

    # Current color
    ccol = colors[ii] 

    # Current label
    clabel = labels[ii]
 
    # Read the file
    extension,lat,lon,eamp,epha,namp,npha,vamp,vpha = read_convolution_file.main(cfile)

    # Combine amp and pha
    eamp = np.multiply(eamp,np.cos(epha*(np.pi/180.)))
    namp = np.multiply(namp,np.cos(npha*(np.pi/180.)))
    vamp = np.multiply(vamp,np.cos(vpha*(np.pi/180.)))
 
    # Sort by Latitude
    lat_idx = np.argsort(lat)
    lat = lat[lat_idx]
    lon = lon[lat_idx]
    extension = extension[lat_idx]
    eamp = eamp[lat_idx]
    namp = namp[lat_idx]
    vamp = vamp[lat_idx]

    # Colat?
    xlab = ("Latitude (deg)")
    if (colat == True):
        lat = 90-lat
        xmax2 = 90.-xmin
        xmin2 = 90.-xmax
        xminp = xmin2
        xmaxp = xmax2
        xlab = ("Angular Distance from Load Center (deg)")
        order = np.argsort(lat)
        lat = lat[order]
        lon = lon[order]
        eamp = eamp[order]
        namp = namp[order]
        vamp = vamp[order]

    # Plot
    plot_fig = True
    fs = 10
    plt.subplot(2,1,1)
    plt.plot(lat,namp,color=ccol,linestyle='-',label=clabel)
    plt.title(plot_title,fontsize=fs)
    plt.ylabel('Lateral Displ. (mm)',fontsize=fs)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.legend(loc=1,fontsize=6)
    plt.xlim(xminp,xmaxp)
    plt.ylim(yminh,ymaxh)
    plt.subplot(2,1,2)
    plt.plot(lat,vamp,color=ccol,linestyle='-')
    #plt.title(plot_title,fontsize=fs)
    plt.ylabel('Up Displ. (mm)',fontsize=fs)
    plt.xlabel(xlab,fontsize=fs)
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.xlim(xminp,xmaxp)
    plt.ylim(yminu,ymaxu)
    plt.tight_layout()
    plt.savefig(plot_name_NU,format='pdf')

# show figure
plt.show()

# Let us know when run is complete
print('finished')


