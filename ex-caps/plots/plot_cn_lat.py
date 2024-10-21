# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# Import Python Modules
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from CONVGF.utility import read_convolution_file

#### USER INPUTS ####

# Convolution files:
# Uncomment ONE planetary model (pmod) below
pmod = "PREM"; pmod_short = "PREM"
#pmod = "Homogeneous_Vp05.92_Vs03.42_Rho03.00"; pmod_short = "HomSph"
#pmod = "Homogeneous_Vp05.92_Vs03.42_Rho03.00_nonGrav"; pmod_short = "HomSph-noGrav"
# Use results with a linearly tapered load?
taper = True
# Select the results to plot based on taper and planetary model
if taper:
    #### With Linear Taper
    #### Uncomment ONE "mod" line below (based on any extra details of the results, such as mesh parameters)
    mod = ("LandAndOceans_linear-taper-5_ce_convgf_custom_symcaps_" + pmod + "_stationMesh_z111_z215_z320_z430_azm0p5")
    #mod = ("LandAndOceans_linear-taper-5_ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.001-11.0_0.005-15.0_0.01-20.0_0.1-30.0_0.5-150.0_0.1-160.0_0.01-165.0_0.005-169.0_0.001")
    #mod = ("LandAndOceans_linear-taper-5_ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.5_0.5_60.0_90.0_0.1_0.5_70.0_90.0_0.01_0.5_75.0_90.0_0.005_0.5_79.0_90.0_0.001_0.5")
    plot_title = (pmod_short + " | Polar Caps, Radius 10deg, 5deg Taper | 1m Fresh Water")
else: 
    #### No Taper
    #### Uncomment ONE "mod" line below (based on any extra details of the results, such as mesh parameters)
    mod = ("LandAndOceans_no-taper_ce_convgf_custom_symcaps_" + pmod + "_stationMesh_z111_z215_z320_z430_azm0p5")
    #mod = ("LandAndOceans_no-taper_ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.001-11.0_0.005-15.0_0.01-20.0_0.1-30.0_0.5-150.0_0.1-160.0_0.01-165.0_0.005-169.0_0.001")
    #mod = ("LandAndOceans_no-taper_ce_convgf_custom_symcaps_" + pmod + "_commonMesh_0.5_0.5_60.0_90.0_0.1_0.5_70.0_90.0_0.01_0.5_75.0_90.0_0.005_0.5_79.0_90.0_0.001_0.5")
    plot_title = (pmod_short + " | Polar Caps, Radius 10deg, No Taper | 1m Fresh Water")

# Scope of Figure
xmin = -90.0
xmax = 90.0
# Use Co-Latitude?
colat = False

# Filenames 
cfile = ("../utility/output/cn_" + mod + ".txt")
plot_name = ("./output/" + mod + ".pdf")
plot_name_NU = ("./output/" + mod + "_NU.pdf")

####################

#### BEGIN CODE ####

# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
outdir = "./output/"

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
if colat:
    lat = 90-lat
    xmax2 = 90.-xmin
    xmin2 = 90.-xmax
    xmin = xmin2
    xmax = xmax2
    xlab = ("Angular Distance from Load Center (deg)")
    print(xmin)
    print(xmax)
    order = np.argsort(lat)
    lat = lat[order]
    lon = lon[order]
    eamp = eamp[order]
    namp = namp[order]
    vamp = vamp[order]
            
# Plot
plot_fig = True
fs = 10
if plot_fig:
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(lat,eamp,color='black')
    plt.title('East | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.xlim(xmin,xmax)
    plt.grid(True)
    plt.subplot(3,1,2)
    plt.plot(lat,namp,color='black')
    plt.title('North | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.grid(True)
    plt.xlim(xmin,xmax)
    plt.subplot(3,1,3)
    plt.plot(lat,vamp,color='black')
    plt.title('Up | ' + plot_title,fontsize=fs)
    plt.ylabel('Displ. (mm)',fontsize=fs)
    plt.xlabel(xlab,fontsize=fs)
    plt.grid(True)
    plt.xlim(xmin,xmax)
    plt.tight_layout()
    plt.savefig(plot_name,format='pdf')
    plt.show()

    plt.close()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(lat,namp,color='black')
    plt.title(plot_title,fontsize=fs)
    plt.ylabel('Lateral Displacement (mm)',fontsize=fs)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.xlim(xmin,xmax)
    plt.subplot(2,1,2)
    plt.plot(lat,vamp,color='black')
    #plt.title(plot_title,fontsize=fs)
    plt.ylabel('Up Displacement (mm)',fontsize=fs)
    plt.xlabel(xlab,fontsize=fs)
    plt.minorticks_on()
    plt.grid(True,which='both')
    plt.xlim(xmin,xmax)
    plt.tight_layout()
    plt.savefig(plot_name_NU,format='pdf')
    plt.show()

