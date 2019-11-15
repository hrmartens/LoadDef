# *********************************************************************
# FUNCTION TO CONVERT AMPLITUDE AND PHASE TO A PARTICLE MOTION ELLIPSE
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

import numpy as np
from math import pi

# CONVERT E/N/V Amplitude and Phase to Horizontal Particle Motion Ellipse #
# See Section 2.7 of Martens Ph.D. Thesis #

def main(eamp,epha,namp,npha):

    # Compute Angle Between North and East Phases
    delta = np.multiply(epha,(pi/180.)) - np.multiply(npha,(pi/180.))
    # Compute Angle of Rotation Between E/N and the Semimajor Axis of Ellipse
    theta = 0.5*np.arctan2(2.*eamp*namp*np.cos(delta),(np.square(eamp)-np.square(namp)))

    # Determine the Semi-major and Semi-Minor Axes 
    smmjr = np.empty((len(eamp),))
    smmnr = np.empty((len(eamp),))
    for ii in range(0,len(eamp)):
        mydelta = delta[ii]
        mytheta = theta[ii]
        myeamp = eamp[ii]
        myepha = epha[ii]
        mynamp = namp[ii]
        mynpha = npha[ii]
        if (myeamp == 0.):
            smmjr[ii] = mynamp
            smmnr[ii] = 0.000001 # Some Very Small Value (For GMT Plotting)
            if (mynamp == 0.): 
                smmjr[ii] = 0.
                smmnr[ii] = 0.
                theta[ii] = 0.
        elif (mynamp == 0.):
            smmjr[ii] = myeamp
            smmnr[ii] = 0.000001 # Some Very Small Value (For GMT Plotting)
        elif ((mydelta == 0.) | (abs(mydelta) == pi)):
            smmjr[ii] = np.sqrt(myeamp**2 + mynamp**2)
            smmnr[ii] = 0.000001 # Some Very Small Value (For GMT Plotting)
        else:
            # Compute 1/a^2 and 1/b^2 Ellipse Parameters, in the Rotated Coordinate Frame (Aligned with Ellipse)
            coef1 = (1./(np.square(np.sin(mydelta))))*((np.square(np.cos(mytheta))/np.square(myeamp)) - \
                (2.*np.cos(mytheta)*np.sin(mytheta)*np.cos(mydelta))/(np.multiply(myeamp,mynamp)) + \
                (np.square(np.sin(mytheta))/np.square(mynamp)))
            coef2 = (1./(np.square(np.sin(mydelta))))*((np.square(np.sin(mytheta))/np.square(myeamp)) + \
                (2.*np.cos(mytheta)*np.sin(mytheta)*np.cos(mydelta))/(np.multiply(myeamp,mynamp)) + \
                (np.square(np.cos(mytheta))/np.square(mynamp)))
            # Compute the Semimajor and Semiminor Axes (a and b)
            smmjr[ii] = np.sqrt(np.divide(1.,coef1))
            smmnr[ii] = np.sqrt(np.divide(1.,coef2))
 
    # Convert the Angle of Rotation, Theta, to Degrees
    theta_deg = np.degrees(theta)

    # Return Ellipse Parameters
    return smmjr,smmnr,theta_deg

