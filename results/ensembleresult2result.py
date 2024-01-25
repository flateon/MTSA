import numpy as np
import pandas as pd

ensemble_data = pd.read_csv('results/tst_ett_w_v2/ensemble_20.csv')
groups = ensemble_data.groupby(['dataset', 'pred_len'])

min_val_mse_groups = groups.apply(lambda group: group[group['val_mse'] == group['val_mse'].min()])
ensemble_result = min_val_mse_groups[['dataset', 'pred_len', 'mse', 'mae']]
ensemble_result['method'] = r'{ensemble\\ opt. $\alpha$}'

ensemble_fix_result = groups.apply(lambda group: group[group['alpha'] == 0.75])[
    ['dataset', 'pred_len', 'mse', 'mae']].reset_index(drop=True)
ensemble_fix_result['method'] = r'{ensemble \\ $\alpha=0.75$}'

pretrain_result = groups.apply(lambda group: group[group['alpha'] == 0])[
    ['dataset', 'pred_len', 'mse', 'mae']].reset_index(drop=True)
pretrain_result['method'] = 'pretrain'

finetune_result = groups.apply(lambda group: group[group['alpha'] == 1])[
    ['dataset', 'pred_len', 'mse', 'mae']].reset_index(drop=True)
finetune_result['method'] = 'finetune'

linearprobe_data = pd.read_csv('results/tst_ett_w_v2/linearprobe.csv')
linearprobe_result = linearprobe_data[['dataset', 'pred_len', 'mse', 'mae']]
linearprobe_result['method'] = 'linear probe'

single_data = pd.read_csv('results/test_model_amp.csv')
single_result = single_data[['dataset', 'pred_len', 'mse', 'mae']]
single_result['method'] = 'single'

all_results = pd.concat(
    [ensemble_fix_result, ensemble_result, finetune_result, linearprobe_result, pretrain_result, single_result])

all_results.to_csv('results/tst_ett_w_v2/all_results.csv', index=False)
