import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pyart
import glob
import multiprocessing
import requests
import shutil


import wradlib as wrl
import gzip
import os
from osgeo import osr

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import fiona
import shapely
import pickle

# A def that emails me when the data is ready
import smtplib
from email.message import EmailMessage

def email_alert(subject, body, to):
    
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    
    
    user = 'jeremy.corner.alert@gmail.com'
    password = 'jrdakdhxfmvaigsy'
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    
    server.quit()


time1 = datetime.now()

day = '26'
month = '06'
month1 = 'June'
year = '2021'


past = datetime.now()

for hour in range(24):
    if hour <= 9:
        for minute in range(59):
            if minute % 2 == 0:
                if minute <= 8:
                    time = f'0{hour}0{minute}00'
                    url = f'https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/PrecipRate/PrecipRate_00.00_{year}{month}{day}-{time}.grib2.gz'
                    r = requests.get(url, allow_redirects=True)
                    filename = url[-41:]
                    if int(r.headers.get('content-length', None)) > 10000:
                        open(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/{filename}', 'wb').write(r.content)
                    
                else:
                    time = f'0{hour}{minute}00'
                    url = f'https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/PrecipRate/PrecipRate_00.00_{year}{month}{day}-{time}.grib2.gz'
                    r = requests.get(url, allow_redirects=True)
                    filename = url[-41:]
                    if int(r.headers.get('content-length', None)) > 10000:
                        open(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/{filename}', 'wb').write(r.content)
                    
                    
    else:
        for minute in range(59):
            if minute % 2 == 0:
                if minute <= 8:
                    time = f'{hour}0{minute}00'
                    url = f'https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/PrecipRate/PrecipRate_00.00_{year}{month}{day}-{time}.grib2.gz'
                    r = requests.get(url, allow_redirects=True)
                    filename = url[-41:]
                    if int(r.headers.get('content-length', None)) > 10000:
                        open(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/{filename}', 'wb').write(r.content)
                    
                else:
                    time = f'{hour}{minute}00'
                    url = f'https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/PrecipRate/PrecipRate_00.00_{year}{month}{day}-{time}.grib2.gz'
                    r = requests.get(url, allow_redirects=True)
                    filename = url[-41:]
                    if int(r.headers.get('content-length', None)) > 10000:
                        open(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/{filename}', 'wb').write(r.content)
                    
                
               

for pathname in glob.glob(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/*.gz'):
    with gzip.open(f'{pathname}', 'rb') as f_in:
        with open(f'{pathname[0:-3]}', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
current = datetime.now()
print(current - past) 


data = glob.glob(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/*.grib2')
data.sort()
#print(data[][-12:-10])

ds = xr.open_dataset(data[0])


#ds = xr.open_dataset(os.path.join(pathfiles,datafile), engine='cfgrib')


#Subset the data and create grid
#ds_sub = ds.sel(latitude=slice(43,40),longitude=slice(270,273))
ds_sub = ds.sel(latitude=slice(45,36),longitude=slice(266,275))


y_rad = ds_sub['latitude'].values
x_rad = ds_sub['longitude'].values

mesh_x, mesh_y = np.meshgrid(x_rad, y_rad)

# Put the two coordinate grids in one array.

grid_xy = np.zeros((900,900,2))

grid_xy[:,:,0] = mesh_x
grid_xy[:,:,1] = mesh_y

# Read the watershed shapefile.
# this file is in UTM
fname_shp = '/gpfs/fs1/home/ac.jcorner/Rainfall/Data/city_detailed_utm/city_detailed_utm.shp'

dataset, inLayer = wrl.io.open_vector(fname_shp)
borders, keys = wrl.georef.get_vector_coordinates(inLayer, key='node_id')

# Define different projections that will be used in the processing.

proj_wgs = osr.SpatialReference()
proj_wgs.ImportFromEPSG(4326)

proj_aeqd = osr.SpatialReference()
proj_aeqd.ImportFromEPSG(54032)

proj_IL = osr.SpatialReference()
proj_IL.ImportFromEPSG(26771)

proj_IL_UTM = osr.SpatialReference()
proj_IL_UTM.ImportFromEPSG(26916)

# Reproject the radar grid to UTM, to match the shapefile.
grid_xy_utm = wrl.georef.reproject(grid_xy,
                                projection_source=proj_wgs,
                                projection_target=proj_IL_UTM)

x_rad_utm = grid_xy_utm[:,:,0]
y_rad_utm = grid_xy_utm[:,:,1]

# Create a mask to reduce size.
# Reduce grid size using a bounding box (to enhancing performance)
bbox = inLayer.GetExtent()

buffer = 100.
bbox = dict(left=bbox[0] - buffer, right=bbox[1] + buffer,
            bottom=bbox[2] - buffer, top=bbox[3] + buffer)
mask = (((grid_xy_utm[..., 1] > bbox['bottom']) & (grid_xy_utm[..., 1] < bbox['top'])) &
        ((grid_xy_utm[..., 0] > bbox['left']) & (grid_xy_utm[..., 0] < bbox['right'])))

# Create vertices for each grid cell
# (MUST BE DONE IN NATIVE COORDINATES)

grdverts = wrl.zonalstats.grid_centers_to_vertices(x_rad_utm[mask],
                                                   y_rad_utm[mask], 824,
                                                   824)

past = datetime.now()

file_hour = '00'
grid_p = np.zeros((900,900))

for file in data:  
    if file[-12:-10] == file_hour:
        df = xr.open_dataset(file)
        p_amount = df.sel(latitude=slice(45,36),longitude=slice(266,275))['unknown'].values
        grid_p[:,:] = p_amount/30.0 + grid_p[:,:]
    
    else:
        pd.DataFrame(grid_p).to_csv(f"/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/one_hour_{file_hour}_agg.csv")
        file_hour = file[-12:-10]
        grid_p = np.zeros((900,900))
        df = xr.open_dataset(file)
        p_amount = df.sel(latitude=slice(45,36),longitude=slice(266,275))['unknown'].values
        grid_p[:,:] = p_amount/30.0 + grid_p[:,:]
              
pd.DataFrame(grid_p).to_csv(f"/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/one_hour_{file_hour}_agg.csv")        
current = datetime.now()
print(current - past)



csv = glob.glob(f'/gpfs/fs1/home/ac.jcorner/Rainfall/Data/MRMS/{day}_{month1}_{year}/*agg.csv')
csv.sort()


def data_agg(data_file_path):
    
    """this is a function built so that MRMS data can be ran in parallel.
    
        data_file_path = the path for the data file
        
        returns csv that aggregates the MRMS file into the catchments."""
    
    past = datetime.now()
    df = pd.read_csv(data_file_path)
    df = df.to_numpy()
    df = df[:,1:]
    
    MRMS_data_agg = []
    
    hour = data_file_path[-10:-8]

    with fiona.open("/gpfs/fs1/home/ac.jcorner/Rainfall/Data/city_detailed_utm/city_detailed_utm.shp") as c:
            for record in c:
                amount = []
                shape = shapely.geometry.shape(record['geometry'])



                for count in range(y_rad_utm[1].size):
                    for i in y_rad_utm[count]:
                        if i >= bbox['bottom'] and i <= bbox['top'] :
                            for j in x_rad_utm[count]:
                                if j >= bbox['left'] and j <= bbox['right']:
                                    point = shapely.geometry.Point(j, i)
                                    if shape.contains(point):
                                        x = np.where(x_rad_utm[count] == j)
                                        y = np.where(y_rad_utm[count] == i)
                                        amount.append(df[x[0],y[0]][0])
                                        

                if len(amount) > 0:
                    maxi = np.max(amount)
                    print(f"catchment {record['id']} with the amount of {maxi} at {hour}.")
                    MRMS_data_agg.append(maxi)

                else:
                    print(f"catchment {record['id']} with the amount of nan at {hour}.")
                    f_amount = np.average(amount)
                    MRMS_data_agg.append(f_amount)

    f = np.array(MRMS_data_agg)
    df = pd.DataFrame(f)
    
    df.to_csv(f'/gpfs/fs1/home/ac.jcorner/Rainfall/MRMS_data_agg_{year}{month}{day}_{hour}.csv')
    
    email_alert('Done, Time To Talk To Scott', f'the script should be done now for {hour}', 'jeremy.corner1998@gmail.com')
    
    current = datetime.now()
    print(current - past)
    
    return MRMS_data_agg

for timestamp in range(len(csv)):
    d = multiprocessing.Process(target=data_agg, args=[csv[timestamp]])
    d.start()

time2 = datetime.now()
print(time2 - time1)

