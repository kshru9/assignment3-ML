import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

np.random.seed(42)
data = load_digits()

plt.figure()

pca = PCA(n_components=2)
proj = pca.fit_transform(data.data)
plt.scatter(proj[:, 0], proj[:, 1], c=data.target, cmap="Paired")
plt.colorbar()
plt.savefig('./figures/PCA_digits.png',dpi=400)