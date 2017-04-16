
print('Reading in data...')

import pandas as pd
import numpy as np

all_df = pd.read_csv('all_df.csv')



print('Performing PCA...')

from sklearn.decomposition import PCA

num_comp = 3
comp_axes = []
for i in range(0,num_comp):
    comp_axes.append('PC'+str(i+1))

pca = PCA(n_components=num_comp)
pca_df = all_df
pca_df.index = pca_df.id
pca_df = pca_df.drop(['id', 'train/test','location','fault_severity'],axis=1)
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

ind = random.sample(range(0, a.shape[0]), 6000)
a2 = a[ind,:]
ind_oth = list(set(range(0, a.shape[0])) - set(ind))
a3 = a[ind_oth,:]
print("    calculating linkage (this make take a while)")
z = hac.linkage(a2, method='complete')
knee = np.diff(z[::-1, 2], 2)
num_clust = knee.argmax() + 2
#num_clust = 7
part = hac.fcluster(z, num_clust, 'maxclust')
clr = ['#2200CC','#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
       '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC']

# can't find leaders with hac :(
# instead, take each location and find the category of its closest labeled neighbour
part_oth = np.zeros([a3.shape[0], 1])
# take cluster centres and define value to all locations by simple distance
for idx, location in enumerate(a3):
    #x = location[0]
    #y = location[1]
    #part_diffs = abs(a2[:,0]-x) + abs(a2[:,1] - y)
    part_diffs = np.zeros([a2.shape[0],1])
    for i in range(0, num_comp):
        part_diffs = part_diffs + np.reshape(abs(a2[:,i]-location[i]),[a2.shape[0],1])
    match_ind = part_diffs.argmin()
    part_oth[idx] = part[match_ind]


# we now have a list of PCs (a2/a3) and their categories (part, part_oth), we need to match back to their ids
location_categories = pca_2d_df
location_categories['category'] = 0
location_categories = location_categories.reset_index(drop=False)
#location_categories = location_categories.reset_index(drop=True)
location_categories['category'][ind] = part
location_categories['category'][ind_oth] = part_oth

np.save('location_categories_3_6',location_categories)
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for cluster in set(part):
    ax.scatter(location_categories['PC1'][location_categories['category'] == cluster],
                location_categories['PC2'][location_categories['category'] == cluster],
                location_categories['PC3'][location_categories['category'] == cluster],color=clr[cluster])


plt.show()
"""
"""
for cluster in set(part):
    plt.scatter(location_categories['PC_1'][location_categories['category'] == cluster],
                location_categories['PC_2'][location_categories['category'] == cluster], color=clr[cluster])


plt.show()
"""

"""
fig, axes23 = plt.subplots(2, 3)

for method, axes in zip(['single', 'complete'], axes23):
    z = hac.linkage(a, method=method)

    # Plotting
    axes[0].plot(range(1, len(z)+1), z[::-1, 2])
    knee = np.diff(z[::-1, 2], 2)
    axes[0].plot(range(2, len(z)), knee)

    num_clust1 = knee.argmax() + 2
    knee[knee.argmax()] = 0
    num_clust2 = knee.argmax() + 2

    axes[0].text(num_clust1, z[::-1, 2][num_clust1-1], 'possible\n<- knee point')

    part1 = hac.fcluster(z, num_clust1, 'maxclust')
    part2 = hac.fcluster(z, num_clust2, 'maxclust')

    clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    '#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']

    for part, ax in zip([part1, part2], axes[1:]):
        for cluster in set(part):
            ax.scatter(a[part == cluster, 0], a[part == cluster, 1],
                       color=clr[cluster])

    m = '\n(method: {})'.format(method)
    plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
             ylabel='{}\ncluster distance'.format(m))
    plt.setp(axes[1], title='{} Clusters'.format(num_clust1))
    plt.setp(axes[2], title='{} Clusters'.format(num_clust2))


plt.tight_layout()
plt.show()
"""


