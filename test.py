import hopper
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

rgauss = np.random.normal(size=(1000,2))

h = hopper.hopper(rgauss)
h.hop(10)
h.hop(20)
print(h.path)
print(h.vcells)

# plt.scatter(rgauss[:,0],rgauss[:,1],c=h.vcells)
# plt.scatter(rgauss[h.path,0],rgauss[h.path,1])
# plt.show()
rgauss = sc.AnnData(rgauss)

smaller = hopper.compress(h, rgauss)
print(smaller)
print(smaller.obs)
plt.scatter(smaller.X[:,0],smaller.X[:,1])
plt.show()
