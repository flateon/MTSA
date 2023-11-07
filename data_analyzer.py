import pandas as pd

if __name__ == '__main__':
    results_df = pd.read_csv('./model_metrics.csv')

    print("\nMetrics of Each Model under Their Best Transform:")
    best_transforms = results_df.groupby(['dataset', 'model'], sort=False)['mae'].idxmin()
    # Display the metrics of each model under their best transform
    with_best_trans = results_df.loc[best_transforms][
        ['dataset', 'model', 'transform', 'mse', 'mae', 'mape', 'smape', 'mase']]
    print(with_best_trans)
    with_best_trans.to_csv('with_best_trans.csv', index=False, float_format='%.3g')
    # Display avg metrics of each model under their best transform over all datasets

    print("\nAvg Metrics of Each Model under Their Best Transform over All Datasets:")
    avg_metric_df = with_best_trans[['model', 'mse', 'mae', 'mape', 'smape', 'mase']].groupby(['model'],
                                                                                              sort=False).mean().reset_index()
    avg_metric_df.to_csv('avg_metrics.csv', index=False, float_format='%.3g')
    print(avg_metric_df)

    # Display the best transform for each models over all datasets
    print("\nBest Transform for Each Model over All Datasets:")
    avg_over_all_datasets = results_df.groupby(['model', 'transform'], sort=False)['mae'].mean().reset_index()
    best_transforms_idx = avg_over_all_datasets.groupby(['model'], sort=False)['mae'].idxmin()
    best_transforms = avg_over_all_datasets.loc[best_transforms_idx][['model', 'transform']]
    print(best_transforms)

    # Display the LR and EMA model
    print("\nLR and EMA Model:")
    idx = (results_df['model'] == 'LinearRegression') | (results_df['model'] == 'ExponentialSmoothing')
    lr_ema_df = results_df.loc[idx]
    lr_ema_df.to_csv('lr_ema.csv', index=False, float_format='%.3g')
    print(lr_ema_df)
