
print('Reading in data...')

import pandas as pd
import numpy as np

all_df = pd.read_csv('all_df.csv')
loc_id_map = pd.DataFrame()
loc_id_map['location'] = all_df['location']
loc_id_map.index = all_df['id']
loc_all_df = all_df.groupby('location').sum().drop('id',axis=1)


print('Performing PCA...')

from sklearn.decomposition import PCA

num_comp = 2
comp_axes = []
for i in range(0,num_comp):
    comp_axes.append('PC'+str(i+1))

pca = PCA(n_components=num_comp)
pca_df = loc_all_df
#pca_df.index = pca_df.id
pca.fit(pca_df)

pca_2d = pca.transform(pca_df)
pca_2d_df = pd.DataFrame(pca_2d)
pca_2d_df.index = pca_df.index
pca_2d_df.columns = comp_axes

print(pca.explained_variance_ratio_)

# matplotlib inline
"""
ax = pca_2d_df.plot(kind='scatter', x='PC_2', y='PC_1', figsize=(16,8))

for i, id_x in enumerate(pca_df.index):
    ax.annotate(
        id_x,
        (pca_2d_df.iloc[i].PC_2, pca_2d_df.iloc[i].PC_1)
    )
"""



print("Applying K-Means Clustering...")
import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D

#a = np.load('pca.npy')
a = pca_2d_df.values
location_ind = pca_2d_df.index

print("    calculating linkage (this make take a while)")
z = hac.linkage(a, method='complete')
knee = np.diff(z[::-1, 2], 2)
num_clust = knee.argmax() + 2
#num_clust = 7
part = hac.fcluster(z, num_clust, 'maxclust')
clr = ['#2200CC','#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
       '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC']

# we now have a list of PCs (a2/a3) and their categories (part, part_oth), we need to match back to their ids
location_categories = pca_2d_df
location_categories['category'] = part
#location_categories = location_categories.reset_index(drop=False)
location_categories.to_csv('k_means_loc_2d.csv')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
"""
for cluster in set(part):
    plt.scatter(location_categories['PC1'][location_categories['category'] == cluster],
                location_categories['PC2'][location_categories['category'] == cluster],
                location_categories['PC3'][location_categories['category'] == cluster],color=clr[cluster])


plt.show()
"""


"""
for cluster in set(part):
    plt.scatter(location_categories['PC1'][location_categories['category'] == cluster],
                location_categories['PC2'][location_categories['category'] == cluster], color=clr[cluster])


plt.show()
"""



