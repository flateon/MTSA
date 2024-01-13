import pandas as pd

results = pd.read_csv('results/test_model_with_timesnet.csv')

metrics = ['MSE', 'MAE']
title = 'Models'
results_title = 'model'
# models = results[results_title].unique()
models = ['FLinear', 'TimesNet', 'Linear', 'DLinear', 'ARIMA']


form = [[title, ''], ['Metric', ''], ]
results = results[results[results_title].isin(models)]

for m in models:
    form[0] += [m] * len(metrics)
    form[1] += metrics

for dataset in results['dataset'].unique():
    for pred_len in results['pred_len'].unique():
        row = [dataset, str(pred_len)]
        for m in models:
            data = results[(results['dataset'] == dataset) &
                           (results['pred_len'] == pred_len)]
            for metric in metrics:
                value = data[data[results_title] == m][metric.lower()].values[0]
                if value == sorted(data[metric.lower()])[0]:
                    row.append(r"\textcolor{red}{\textbf{" + f'{value:.3f}' + r"}}")
                elif value == sorted(data[metric.lower()])[1]:
                    row.append(r"\underline{\textcolor{blue}{" + f'{value:.3f}' + r"}}")
                else:
                    row.append(f'{value:.3f}')
        form.append(row)

    avg_row = [dataset, 'Avg']
    data = results[(results['dataset'] == dataset)]
    for m in models:
        for metric in metrics:
            all_model_avg = sorted(
                [data[data[results_title] == all_m][metric.lower()].values.mean() for all_m in models])
            value = data[data[results_title] == m][metric.lower()].values.mean()
            if value == all_model_avg[0]:
                avg_row.append(r"\textcolor{red}{\textbf{" + f'{value:.3f}' + r"}}")
            elif value == all_model_avg[1]:
                avg_row.append(r"\underline{\textcolor{blue}{" + f'{value:.3f}' + r"}}")
            else:
                avg_row.append(f'{value:.3f}')
    form.append(avg_row)
pd.DataFrame(form[1:], columns=form[0]).to_csv('./results/test_form.csv', index=False)
print(' \\\ \n'.join([' & '.join(f) for f in form]))
