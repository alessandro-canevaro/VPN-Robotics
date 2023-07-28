import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from PIL import Image

raw_data = np.genfromtxt("fin_values2.csv", delimiter=",", dtype=np.float16)
raw_data[0, 0] = 0.0

pathloss = raw_data.copy()
pathloss[pathloss == 0] = np.nan

thrs = -95
thrs_pathloss = raw_data.copy()
thrs_pathloss[thrs_pathloss == 0] = -200
thrs_pathloss[(thrs_pathloss > thrs)] = 0
thrs_pathloss[thrs_pathloss <= thrs] = 1

#plots
plt.subplot(1, 2, 1)
im = plt.imshow(pathloss, cmap='inferno', interpolation='nearest')
plt.colorbar(im,fraction=0.046, pad=0.04)

plt.subplot(1, 2, 2)
cmap = mpl.colors.ListedColormap(['w', 'k'])
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
im = plt.imshow(thrs_pathloss, interpolation='nearest', cmap=cmap, norm=norm)

plt.tight_layout()
plt.grid()
plt.savefig("heatmap.pdf")
print("plotted")

#save map
img = Image.fromarray(((thrs_pathloss-1)**2*255).astype('uint8'))
imgrgb = img.convert('RGB')
imgrgb.save('blocked_map.png')