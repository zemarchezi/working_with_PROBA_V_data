#%%
import numpy as np
import pandas as pd
import gzip
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pyIGRF

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

import itertools
import matplotlib.patheffects as pe
from matplotlib.patheffects import Stroke, Normal

import cartopy.mpl.geoaxes

def calculateMag(xspace, yspace, year, height):
    euator = np.zeros((len(xspace), 2))
    inclination = np.zeros((len(yspace), len(xspace)))
    magnt = np.zeros((len(yspace), len(xspace)))
    for x in range(len(xspace)):
        for y in range(len(yspace)):
            decl, inc, hMag, xMag, yMag, zMag, fMAg = pyIGRF.igrf_value(yspace[y], xspace[x], height, year)
            inclination[y,x] = inc
            magnt[y,x] = fMAg



    
    equator = []
    for ii in range(inclination.shape[1]):
        
        temp = inclination[:,ii]

        sts = np.where((temp > -1) & (temp < 1))
        # print(sts)

        idx = sts[0][np.argmin(abs(temp[sts]))]


        equator.append(yspace[idx])    

    return inclination, equator, magnt

def zebra_frame(self, lw=3, crs=None, zorder=None, iFlag_outer_frame_in = None):    
    # Alternate black and white line segments
    bws = itertools.cycle(["k", "w"])
    self.spines["geo"].set_visible(False)
    
    if iFlag_outer_frame_in is not None:
        #get the map spatial reference        
        left, right, bottom, top = self.get_extent()
        crs_map = self.projection
        xticks = np.arange(left, right+(right-left)/9, (right-left)/8)
        yticks = np.arange(bottom, top+(top-bottom)/9, (top-bottom)/8)
        #check spatial reference are the same           
        pass
    else:        
        crs_map =  crs
        xticks = sorted([*self.get_xticks()])
        xticks = np.unique(np.array(xticks))        
        yticks = sorted([*self.get_yticks()])
        yticks = np.unique(np.array(yticks))        

    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[yticks[0], yticks[0]], [yticks[-1], yticks[-1]]]
            else:
                xs = [[xticks[0], xticks[0]], [xticks[-1], xticks[-1]]]
                ys = [[start, end], [start, end]]

            # For first and last lines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for (xx, yy) in zip(xs, ys):
                self.plot(xx, yy, color=bw, linewidth=max(0, lw - self.spines["geo"].get_linewidth()*2), clip_on=False,
                    transform=crs_map, zorder=zorder, solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        Stroke(linewidth=lw, foreground="black"),
                        Normal(),
                    ],
                )
#%%

xspace = np.arange(-180,181,0.5)
yspace = np.arange(-90,91, 0.5)
incl, euator, magnt = calculateMag(xspace, yspace, 2024., 100)

  # Print each line (or process it as needed)
# %%

day1 = 1
day2 = 10

datas = []

columns = ['Y', 'M', 'D', 'H', 'MI', 'S', 'mS', 'AMJD', 'FLAG', 'e-fl-00', 'e-fl-01', 'e-fl-02', 'e-fl-03', 
 'e-fl-04', 'e-fl-05', 'p-fl-00', 'p-fl-01', 'p-fl-02', 'p-fl-03', 'p-fl-04', 'p-fl-05', 'p-fl-06', 
 'p-fl-07', 'p-fl-08', 'p-fl-09', 'He-fl-00', 'He-fl-01', 'He-fl-02', 'He-fl-03', 'He-fl-04', 
 'He-fl-05', 'He-fl-06', 'He-fl-07', 'He-fl-08', 'He-fl-09', 'e-dfl-00', 'e-dfl-01', 'e-dfl-02', 
 'e-dfl-03', 'e-dfl-04', 'e-dfl-05', 'p-dfl-00', 'p-dfl-01', 'p-dfl-02', 'p-dfl-03', 'p-dfl-04', 
 'p-dfl-05', 'p-dfl-06', 'p-dfl-07', 'p-dfl-08', 'p-dfl-09', 'He-dfl-00', 'He-dfl-01', 'He-dfl-02', 
 'He-dfl-03', 'He-dfl-04', 'He-dfl-05', 'He-dfl-06', 'He-dfl-07', 'He-dfl-08', 'He-dfl-09', 
 'e-Nit', 'p-Nit', 'He-Nit', 'e-Chi2', 'p-Chi2', 'He-Chi2', 'Pitch', 'B', 'Bvec-0', 'Bvec-1', 
 'Bvec-2', 'Long', 'Lat', 'Rad', 'PitchU', 'BvecU-0', 'BvecU-1', 'BvecU-2', 'BU', 'LU', 'Rinv', 
 'Lat_mag', 'Lat_inv', 'MLTU', 'PitchI', 'BvecI-0', 'BvecI-1', 'BvecI-2', 'BI', 'LI', 'MLTI']
for day in range(day1,day2+1):
    PATH = f'/Users/jose/Downloads/PROBAV_EPT_PersonalDataSet/PROBAV_EPT_202405{day:02d}_L1d.dat.gz'
    data = np.loadtxt(PATH,skiprows=25)

    df = pd.DataFrame(data, columns=columns)

    df['datetime'] = pd.to_datetime(
        df[['Y', 'M', 'D', 'H', 'MI', 'S']].astype(int).astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H-%M-%S'
    )
    df['datetime'] += pd.to_timedelta(df['mS'] / 1000, unit='s')


    datas.append(df)


#%%

dados = pd.concat(datas)

# Assuming your DataFrame is called df and it has 'e-fl-00', 'Long', and 'Lat' columns
longitudes = dados['Long']
latitudes = dados['Lat']
values = dados['e-fl-00']

# Set up the map
fig = plt.figure(figsize=(14, 14), dpi=200)

# crs = ccrs.Orthographic(central_longitude=-45, central_latitude=-5)
crs = ccrs.PlateCarree(central_longitude=0,)


ax = fig.add_subplot(projection=crs)

ax.coastlines(resolution='110m')
# ax.set_extent((-180, -180, -45, 5))
gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
# gl.top_labels = False  # Remove top labels
# gl.right_labels = False  # Remove right labels
# gl.xformatter = None  # Optional: Remove formatting for longitude
# gl.yformatter = None
gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 20))  # Longitudes every 10 degrees
gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 20))


# gl = ax.gridlines(crs=crs, draw_labels=False,
#                   linewidth=2, color='gray', alpha=0.5, linestyle='--')

# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# m = Basemap(projection='cyl', resolution='c',
#             llcrnrlat=-90, urcrnrlat=10,
#             llcrnrlon=-110, urcrnrlon=-30, ax=ax)
# zebra_frame(ax, crs=crs)
# Draw map features
# m.drawcoastlines()
# m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])

# Scatter plot of the data
sc = ax.scatter(longitudes, latitudes, c=values, cmap='jet', norm=LogNorm(), s=4, alpha=0.8)

# ax.plot(xspace,euator, color="magenta",label="Magnetic equator",transform=ccrs.PlateCarree() )

# # levels = np.arange(22000, 30000, 40000)
# CS = ax.contour(xspace, yspace,magnt, cmap='jet', transform=ccrs.PlateCarree())
# ax.clabel(CS, inline=1, fontsize=10)
# # Add a colorbar
cbar = plt.colorbar(sc, pad=0.05, shrink=0.7, orientation='horizontal')
cbar.set_label(r'MeV$^{-1}$cm$^{-2}$s$^{-1}$sr$^{-1}$', fontsize=14)

# Add a title
plt.title(f"500-600 keV, {day1:02d}-{day2:02d} May 2024 \n", fontsize=16)

plt.savefig(f'proba-v_500-600keV-{day1:02d}-{day2:02d}_May_2024.jpg', dpi=200, bbox_inches='tight')

# %%

# %%