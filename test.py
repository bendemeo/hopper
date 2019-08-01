from treehopper.hoppers import hopper
import numpy as np
import matplotlib.pyplot as plt

rgauss = np.random.normal(size=(1000,2))

h = hopper(rgauss)
h.hop(10)
print(h.path)
print(h.vcells)

plt.scatter(rgauss[:,0],rgauss[:,1],c=h.vcells)
plt.scatter(rgauss[h.path,0],rgauss[h.path,1])
plt.show()
