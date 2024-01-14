import pandas as pd

results = pd.read_csv('results/test_model_with_timesnet.csv')

metrics = ['MSE', 'MAE']
col_name = 'Models'
results_col_name = 'model'
# columns = results[results_col_name].unique()
columns = ['FLinear', 'TimesNet', 'Linear', 'DLinear', 'ARIMA']
results = results[results[results_col_name].isin(columns)]

form = [[col_name, ''], ['Metric', ''], ]
for m in columns:
    form[0] += [m] * len(metrics)
    form[1] += metrics

for dataset in results['dataset'].unique():
    for pred_len in results['pred_len'].unique():
        row = [dataset, str(pred_len)]
        for c in columns:
            data = results[(results['dataset'] == dataset) &
                           (results['pred_len'] == pred_len)]
            for metric in metrics:
                value = data[data[results_col_name] == c][metric.lower()].values[0]
                if value == sorted(data[metric.lower()])[0]:
                    row.append(r"\textcolor{red}{\textbf{" + f'{value:.3f}' + r"}}")
                elif value == sorted(data[metric.lower()])[1]:
                    row.append(r"\underline{\textcolor{blue}{" + f'{value:.3f}' + r"}}")
                else:
                    row.append(f'{value:.3f}')
        form.append(row)

    avg_row = [dataset, 'Avg']
    data = results[(results['dataset'] == dataset)]
    for c in columns:
        for metric in metrics:
            all_model_avg = sorted(
                [data[data[results_col_name] == all_c][metric.lower()].values.mean() for all_c in columns])
            value = data[data[results_col_name] == c][metric.lower()].values.mean()
            if value == all_model_avg[0]:
                avg_row.append(r"\textcolor{red}{\textbf{" + f'{value:.3f}' + r"}}")
            elif value == all_model_avg[1]:
                avg_row.append(r"\underline{\textcolor{blue}{" + f'{value:.3f}' + r"}}")
            else:
                avg_row.append(f'{value:.3f}')
    form.append(avg_row)
pd.DataFrame(form[1:], columns=form[0]).to_csv('./results/test_form.csv', index=False)
print(' \\\ \n'.join([' & '.join(f) for f in form]))
