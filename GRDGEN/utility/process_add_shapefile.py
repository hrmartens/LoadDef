# *********************************************************************
# PROGRAM TO GENERATE A SPARSE LAND-SEA MASK AROUND THE ANTARCTIC COASTLINE
# USING SHAPEFILES FROM THE ANTARCTIC DIGITAL DATABASE (ADD)
# http://www.add.scar.org/home/add7
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

# Import Python Modules
from __future__ import print_function
import shapefile
import numpy as np
from pyproj import Proj, transform
from shapely import geometry
import sys
import os
from GRDGEN.utility import make_lsmask_sparse
 
def main(filename_shp, filename_dbf, landsea_field, outfile, resolution):
 
    # Update
    print("Generating Antarctic Land-Sea Mask...")

    # Create Folders
    if not (os.path.isdir("../../output/Land_Sea/")):
        os.makedirs("../../output/Land_Sea/")
    outdir = "../../output/Land_Sea/"
 
    # Set-Up Lat/Lon Grid
    lat = np.arange(-86.,-60.,resolution)
    lon = np.arange(0.,360.,resolution)
    xv,yv = np.meshgrid(lon,lat)
    ilon = xv.flatten()
    ilat = yv.flatten()

    # Read Shapefiles
    print("Reading Shapefiles")
    myshp = open(filename_shp, "rb")
    mydbf = open(filename_dbf, "rb")
    sf = shapefile.Reader(shp=myshp, dbf=mydbf)

    # Extract Shapes
    shapes = sf.shapes()

    # Extract Attributes and Record Values
    attrib = sf.fields[1:]
    field_names = [field[0] for field in attrib]
    lsidx = field_names.index(landsea_field)
    landidx = []; count = 0
    shlfidx = []; oceanidx = []
    rmplidx = []
    for cc in sf.iterRecords():
        if (cc[lsidx] == 'land'):
            landidx.append(count)
        elif ((cc[lsidx] == 'iceshelf') | (cc[lsidx] == 'ice shelf')):
            shlfidx.append(count)
        elif (cc[lsidx] == 'rumple'):
            rmplidx.append(count)
        else:
            oceanidx.append(count)
        count += 1

    # Specify Output (from ADD Manual, lat_ts = standard parallel) Projection
    outProj = Proj(init='epsg:3031') # See spatialreference.org | 3031 for ADD specifically
    #inProj = Proj(proj='stere',lat_0=-90.,lat_ts=-71.,lon_0=0.,ellps='WGS84',datum='WGS84')
 
    # Specify Input Projection
    inProj = Proj(init='epsg:4326')

    # Set Up Initial Array (0=ocean, 1=land)
    land_sea = np.zeros((len(ilon),))

    # Loop Through Grid Points
    print("Looping through Grid Points")
    for kk in range(0,len(ilon)):

        # Number Complete
        print('Number of Grid Points Completed: %6d of %6d' %(kk, len(ilon)))

        # Current Lat/Lon
        clon = ilon[kk]
        clat = ilat[kk]
   
        # Quickly Set Aside Points that are Clearly Land
        if ((clon > 0.) & (clon < 150.)):
            if ((clat > -90.) & (clat < -74.)):
                land_sea[kk] = 1
                continue
        if ((clon > 90.) & (clon < 150.)):
            if ((clat > -90.) & (clat < -70.)):
                land_sea[kk] = 1
                continue
        if ((clon > 240.) & (clon < 270.)):
            if ((clat > -90.) & (clat < -76.)):
                land_sea[kk] = 1
                continue

        # Quickly Set Aside Points that are Clearly Ocean
        if ((clon > 180.) & (clon < 270.)):
            if ((clat > -70.) & (clat < -60.)):
                land_sea[kk] = 0
                continue
        if ((clon > 0.) & (clon < 30.)):
            if ((clat > -68.) & (clat < -60.)):
                land_sea[kk] = 0
                continue
        if ((clon > 330.) & (clon < 360.)):
            if ((clat > -68.) & (clat < -60.)):
                land_sea[kk] = 0
                continue
        if ((clon > 180.) & (clon < 210.)):
            if ((clat > -75.) & (clat < -70.)):
                land_sea[kk] = 0
                continue 

        # Convert Grid Points to Projected Coordinates
        cx,cy = transform(inProj,outProj,clon,clat)

        # Construct the Point in the Point Class
        cpoint = geometry.Point(cx,cy)

        # Loop Through Shapes (Only those over Land)
        for ii in range(0,len(landidx)):

            # Current Land Index
            cidx = landidx[ii]

            # Current Shape
            cshape = shapes[cidx]

            # Attribute Information
            #for name in dir(cshape):
                #if not name.startswith('__'):
                    #print(name)

            # Determine the Bounding Box of the Current Shape
            cbbox = cshape.bbox
            
            # Convert BBox to a Polygon Class Variable
            cbbox = geometry.box(cbbox[0],cbbox[1],cbbox[2],cbbox[3])

            # Test if the Current Point is Within the Current Bounding Box
            if cbbox.contains(cpoint):

                # Extract Indices of Points in Individual Polygons
                if (len(cshape.parts) > 1):
                    for jj in range(1,len(cshape.parts)+1):
                        if (jj == len(cshape.parts)):
                            ptidx = [cshape.parts[jj-1],len(cshape.points)]
                            # Create Polygon from Points
                            cpoly = geometry.Polygon(cshape.points[ptidx[0]:ptidx[1]])
                            # Test if Current Point is WIthin the Current Polygon
                            if cpoly.contains(cpoint):
                                land_sea[kk] = 1
                        else:
                            ptidx = [cshape.parts[jj-1],cshape.parts[jj]-1]
                            # Create Polygon from Points
                            cpoly = geometry.Polygon(cshape.points[ptidx[0]:ptidx[1]])
                            # Test if Current Point is WIthin the Current Polygon
                            if cpoly.contains(cpoint):
                                land_sea[kk] = 1
                else:
                    ptidx = [0,len(cshape.points)] 
                    # Create Polygon from Points
                    cpoly = geometry.Polygon(cshape.points[ptidx[0]:ptidx[1]])
                    # Test if Current Point is WIthin the Current Polygon
                    if cpoly.contains(cpoint):
                        land_sea[kk] = 1
 
    # Only Keep Values Near to the Coastlines (to Save Memory)
    gpoints = 5 # Grid Points within Coastline to Keep
    olat,olon,lsmask = make_lsmask_sparse.main(ilat,ilon,land_sea,gpoints)

    # Write Land-Sea Database to Ascii File
    print("Writing Data to File")
    all_data = np.column_stack((olat,olon,lsmask))
    #f_handle = open((outdir + outfile),'w')
    #np.savetxt(f_handle, all_data, fmt='%f %f %d')
    #f_handle.close()
    np.savetxt((outdir + outfile), all_data, fmt='%f %f %d')

