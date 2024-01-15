import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results/SPIRIT_.csv')
results['components'].fillna('no PCA', inplace=True)

# fig, ax = plt.subplots(dpi=300, layout='constrained')
#
# for dataset in results['dataset'].unique():
#     data = results[results['dataset'] == dataset]
#     n_components_mse = data.groupby('n_components')['mse'].apply('mean')
#     x = n_components_mse.index.values / n_components_mse.index.max()
#     y = n_components_mse.values
#     ax.plot(x, y, label=f'{dataset}')
#
# ax.legend()
# ax.set_xlabel('Percent of Components')
# ax.set_ylabel('MSE')
# ax.grid()
# plt.show()

fig, ax = plt.subplots(dpi=300, layout='constrained')
for dataset in results['dataset'].unique():
    data = results[results['dataset'] == dataset]
    n_components_mse = data.groupby('components')['mse'].apply('mean')
    x = n_components_mse.index.values
    y = n_components_mse.values
    ax.plot(x, y, label=f'{dataset}')

ax.set_xlabel('PCA Components setting')
ax.set_ylabel('MSE')
ax.grid()
ax2 = ax.twinx()
# plt.show()

# fig, ax = plt.subplots(dpi=300, layout='constrained')

for dataset in results['dataset'].unique():
    data = results[results['dataset'] == dataset]
    n_components = data.groupby('components')['n_components'].apply('mean')
    x = n_components.index.values
    y = n_components.values / n_components.max()
    ax2.plot(x, y, ':', label=f'{dataset}', alpha=0.5)

# ax2.grid()
ax2.set_ylabel('Relative Number of Components')
ax.legend()

plt.show()
