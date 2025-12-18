#%%https://swe.ssa.esa.int/web/guest/csr-ept-federated
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

from idl_colorbars import *
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
def add_zebra_frame(ax, lw=2, crs="pcarree", zorder=None):

    ax.spines["geo"].set_visible(False)
    left, right, bot, top = ax.get_extent()
    
    # Alternate black and white line segments
    bws = itertools.cycle(["k", "white"])

    xticks = sorted([left, *ax.get_xticks(), right])
    xticks = np.unique(np.array(xticks))
    yticks = sorted([bot, *ax.get_yticks(), top])
    yticks = np.unique(np.array(yticks))
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[bot, bot], [top, top]]
            else:
                xs = [[left, left], [right, right]]
                ys = [[start, end], [start, end]]

            # For first and lastlines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for (xx, yy) in zip(xs, ys):
                ax.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=lw,
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        pe.Stroke(linewidth=lw + 1, foreground="black"),
                        pe.Normal(),
                    ],
                )
#%%

# xspace = np.arange(-180,181,0.5)
# yspace = np.arange(-90,91, 0.5)
# incl, euator, magnt = calculateMag(xspace, yspace, 2024., 100)

  # Print each line (or process it as needed)
# %%
PATH = "/Users/zemarchezi/sat_data/proba-V/"

day1 = 18
day2 = 19

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
    data_path = f'{PATH}PROBAV_EPT_PersonalDataSet/PROBAV_EPT_201707{day:02d}_L1d.dat.gz'
    data = np.loadtxt(data_path,skiprows=25)

    df = pd.DataFrame(data, columns=columns)

    df['datetime'] = pd.to_datetime(
        df[['Y', 'M', 'D', 'H', 'MI', 'S']].astype(int).astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H-%M-%S'
    )
    df['datetime'] += pd.to_timedelta(df['mS'] / 1000, unit='s')


    datas.append(df)


#%%

dados = pd.concat(datas)

# mask = (dados['datetime'] > '2017-07-18 14:30:00') & (dados['datetime'] <= '2017-07-18 15:50:50')

# # mask = (dados['Lat'] <= 15) & (dados['Lat'] >= -60)
# dados = dados.loc[mask]
# # mask2 = (dados['Long'] >= -90) & (dados['Long'] <= -20)
# dados = dados.loc[mask2]

# Assuming your DataFrame is called df and it has 'e-fl-00', 'Long', and 'Lat' columns
longitudes = dados['Long']
latitudes = dados['Lat']
values = dados['e-fl-00']

# plt.figure(figsize=(10,5))
# plt.plot(np.log(values))

mycmap="rainbow"
# mycmap=getcmap(13)

# Set up the map
fig = plt.figure(figsize=(14, 14), dpi=200)

# crs = ccrs.Orthographic(central_longitude=-45, central_latitude=-5)
crs = ccrs.PlateCarree(central_longitude=0)

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(projection=crs)

ax.coastlines()
ax.set_extent((-180, 180, -90, 90))
# ax.set_xticks((-180, -180, -90, -90))
# ax.set_yticks((25, 30, 35, 40))

add_zebra_frame(ax, crs=crs)

gl = ax.gridlines(crs=crs, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# ax.clabel(CS, inline=True, fontsize=10)#, manual=manual_locations)
# ax.scatter(all_stations["GLongg"].values, all_stations["GLAT"].values, c="red")

# Scatter plot of the data
sc = ax.scatter(longitudes, latitudes, c=values, 
                cmap=mycmap, norm=LogNorm(vmin=1e0, vmax=1e6), s=4, alpha=0.8)

# ax.plot(xspace,euator, color="magenta",label="Magnetic equator",transform=ccrs.PlateCarree() )

# CS = ax.contour(xspace, yspace,magnt, cmap='jet', transform=ccrs.PlateCarree())
# ax.clabel(CS, inline=True, fontsize=10)#, manual=manual_locations)
# # Add a colorbar
cbar = plt.colorbar(sc, pad=0.05, shrink=0.7, orientation='horizontal')
cbar.set_label(r'MeV$^{-1}$cm$^{-2}$s$^{-1}$sr$^{-1}$', fontsize=14)

# Add a title
plt.title(f"500-600 keV, {day1:02d}-{day2:02d} Jul 2025 \n", fontsize=16)

plt.savefig(f'proba-v_500-600keV-{day1:02d}-{day2:02d}_Oct_2025.jpg', dpi=200, bbox_inches='tight')

# %%

# %%