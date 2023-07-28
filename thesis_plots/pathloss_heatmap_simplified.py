import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from PIL import Image
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "arial"


raw_data = np.genfromtxt("fin_values2.csv", delimiter=",", dtype=np.float16)
raw_data[0, 0] = 0.0

pathloss = raw_data.copy()
pathloss[pathloss == 0] = np.nan

cmap = mpl.colors.ListedColormap(['w', 'k'])
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
font_size = 20


#plots
#plt.subplot(1, 2, 1)

thrs = -120
thrs_pathloss = raw_data.copy()
thrs_pathloss[thrs_pathloss == 0] = -200
thrs_pathloss[(thrs_pathloss > thrs)] = 0
thrs_pathloss[thrs_pathloss <= thrs] = 1
im = plt.imshow(thrs_pathloss, interpolation='nearest', cmap=cmap, norm=norm)
im = plt.imshow(pathloss, cmap='inferno', interpolation='nearest')
cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(font_size)
cbar.set_label(label="Pathloss [dBm]", fontsize=font_size+2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(False)
plt.tight_layout()
plt.savefig("heatmap.pdf")
plt.show()
print("plotted")

#plt.subplot(1, 2, 2)
thrs = -120
thrs_pathloss = raw_data.copy()
thrs_pathloss[thrs_pathloss == 0] = -200
thrs_pathloss[(thrs_pathloss > thrs)] = 0
thrs_pathloss[thrs_pathloss <= thrs] = 1
im = plt.imshow(thrs_pathloss, interpolation='nearest', cmap=cmap, norm=norm)

thrs = -95
thrs_pathloss = raw_data.copy()
thrs_pathloss[thrs_pathloss == 0] = -200
thrs_pathloss[(thrs_pathloss > thrs)] = 0
thrs_pathloss[thrs_pathloss <= thrs] = 1
plt.imshow(thrs_pathloss, interpolation='nearest', cmap=cmap, norm=norm, alpha=0.5)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(False)
plt.tight_layout()
plt.savefig("virtual_obs.pdf")
plt.show()
print("plotted")

#save map
img = Image.fromarray(((thrs_pathloss-1)**2*255).astype('uint8'))
imgrgb = img.convert('RGB')
imgrgb.save('blocked_map.png')