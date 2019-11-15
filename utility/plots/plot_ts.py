#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO PLOT TIME SERIES (OUTPUT FROM run_cn.py in TIME SERIES MODE)
# 
# Copyright (c) 2014-2019: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
#
# This file is part of LoadDef.
#
#    LoadDef is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    LoadDef is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LoadDef.  If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************************************

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd() + "/../../")

# Import Python Modules
import numpy as np
from scipy import signal
import datetime
from math import pi
import matplotlib.pyplot as plt

# Input Parameters
station = ("P402")
filename = ("cn_LandOnly_" + station + "_cf_convgf_atml")
#filename = ("cn_OceanOnly_" + station + "_cf_convgf_ntol")
#filename = ("cn_LandAndOceans_" + station + "_cf_convgf_grace")
ts_file = ("../../output/Convolution/" + filename + ".txt")
figname = (filename + ".pdf")

# Detrend the Time Series? (Caution! Assumes equally spaced points)
detrend = True
 
#### Begin Code

# Create Folder
if not (os.path.isdir("./output/")):
    os.makedirs("./output/")
outdir = "./output/"

# Read File
date = np.loadtxt(ts_file,usecols=(0,),dtype='U',unpack=True,skiprows=1)
lat,lon,eamp,epha,namp,npha,vamp,vpha = np.loadtxt(ts_file,delimiter=None,unpack=True,skiprows=1,usecols=(1,2,3,4,5,6,7,8))

# Convert Dates to Datetime Format
mydates = []
hour = 0
for jj in range(0,len(date)):
    mydate = date[jj]
    year = int(mydate[0:4])
    month = int(mydate[4:6])
    day = int(mydate[6:8])
    hour = int(mydate[8:10]) 
    ddate = datetime.datetime(year,month,day,hour,0,0)
    mydates.append(ddate)
# Convert Date List to Numpy Array
mydates = np.array(mydates)

# Convert Amp/Pha to Displacement
ets = np.multiply(eamp,np.cos(np.multiply(epha,(pi/180.))))
nts = np.multiply(namp,np.cos(np.multiply(npha,(pi/180.))))
vts = np.multiply(vamp,np.cos(np.multiply(vpha,(pi/180.))))

# Optionally Detrend
if (detrend == True):
    ets = signal.detrend(ets,type='linear')
    nts = signal.detrend(nts,type='linear')
    vts = signal.detrend(vts,type='linear')
 
# Plot
plt.subplot(3,1,1)
plt.plot_date(mydates,ets,'.',color='black',ms=4,linestyle='-',linewidth=1)
plt.title("Station : " + station)
plt.ylabel('East Disp. (mm)')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.grid(True)
plt.subplot(3,1,2)
plt.plot_date(mydates,nts,'.',color='black',ms=4,linestyle='-',linewidth=1)
plt.ylabel('North Disp. (mm)')
plt.grid(True)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.subplot(3,1,3)
plt.plot_date(mydates,vts,'.',color='black',ms=4,linestyle='-',linewidth=1)
plt.ylabel('Up Disp. (mm)')
plt.xlabel('Date')
plt.xticks(rotation=25)
plt.grid(True)
plt.tight_layout()
plt.savefig((outdir+figname),orientation='portrait',format='pdf')
plt.show()

