#%%
import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results/tst_ett_w_v2/ensemble_20.csv')


# %%
def plot_alpha(results):
    fig, axes = plt.subplots(ncols=3, layout='constrained', dpi=300, figsize=(15, 5))
    # fig.suptitle('Performance of Weight Ensemble model with different alpha', size=16)

    axes[0].set_title('Average MSE of 5 datasets and 4 predict length')
    data = results.groupby('alpha')['mse'].apply('mean')
    pretrain, finetune = data.loc[[0, 1]]
    axes[0].plot(data, marker='o', markersize=4, label='Ensemble model')
    axes[0].plot([0], pretrain, marker='*', markersize=12, label='Pre-trained model')
    axes[0].plot([1], finetune, marker='s', markersize=10, label='Fine-tuned model')
    axes[0].set_xlabel('Alpha')
    axes[0].set_ylabel('Test MSE')
    axes[0].legend()
    axes[0].grid(True)
    ax2 = axes[0].twinx()
    data = results.groupby('alpha')['val_mse'].apply('mean')
    pretrain, finetune = data.loc[[0, 1]]
    ax2.plot(data, marker='o', markersize=4, label='Val MSE', color='C0', alpha=0.3)
    ax2.plot([1], finetune, marker='s', markersize=10, label='Fine-tuned model', color='C2', alpha=0.3)
    ax2.plot([0], pretrain, marker='*', markersize=12, label='Pre-trained', color='C1', alpha=0.3)
    ax2.set_ylabel('Val MSE')
    # ax2.legend()


    axes[1].set_title('Average MSE on 5 datasets')
    for dataset in results['dataset'].unique():
        data = results[results['dataset'] == dataset].groupby('alpha')['mse'].apply('mean')
        data /= data[0]
        axes[1].plot(data, marker='o', markersize=4, label=dataset.split('_')[0])
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Normalized MSE')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title('Average MSE on 4 predict length')
    for predict_length in results['pred_len'].unique():
        data = results[results['pred_len'] == predict_length].groupby('alpha')['mse'].apply('mean')
        data /= data[0]
        axes[2].plot(data, marker='o', markersize=4, label=f'{predict_length}')
    axes[2].set_xlabel('Alpha')
    axes[2].set_ylabel('Normalized MSE')
    axes[2].legend()
    axes[2].grid(True)



plot_alpha(results)
plt.savefig('imgs/weight.pdf')
# plt.show()