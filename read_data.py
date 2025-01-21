#%%
import numpy as np
import pandas as pd
import gzip

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
#%%


  # Print each line (or process it as needed)
# %%

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
for day in range(10,20):
    PATH = f'/Users/jose/Downloads/PROBAV_EPT_PersonalDataSet/PROBAV_EPT_202405{day}_L1d.dat.gz'
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
#%%
# Set up the map
fig, ax = plt.subplots(figsize=(12, 6))
m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, ax=ax)

# Draw map features
m.drawcoastlines()
m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])

# Scatter plot of the data
sc = m.scatter(longitudes, latitudes, c=values, cmap='jet', norm=LogNorm(), s=1, alpha=0.7)

# Add a colorbar
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cbar.set_label(r'MeV$^{-1}$cm$^{-2}$s$^{-1}$sr$^{-1}$')

# Add a title
plt.title("e-fl-00 10-19 May 2024")

# Show the plot
plt.show()
# %%
