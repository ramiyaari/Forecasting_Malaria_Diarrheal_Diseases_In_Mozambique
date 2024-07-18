import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#from datetime import timedelta, datetime
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, rmse
from darts.utils.missing_values import fill_missing_values
from darts.utils.utils import ModelMode, SeasonalityMode

from sklearn.preprocessing import MinMaxScaler

import time
import warnings
import pickle


#Read provinces timeseries data from file
def read_provinces_timeseries_data(filename):
    df = pd.read_csv(filename) 
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y') #format='%Y-%m-%d') #
    df = df.set_index('date')
    return df

#Plots provinces timeseries data
def plot_provinces_timeseries_data(df, provinces, ylabel):
    df = df.reset_index(names=['date'])
    df_long = pd.melt(df,id_vars='date',value_vars=provinces,var_name='province',value_name=ylabel)
    g = sns.FacetGrid(df_long, col="province", col_wrap=4, hue="province", sharey=False, sharex=False, height=3, aspect=1.33)
    g.map(sns.lineplot, "date", ylabel)
    [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

#Return a univartiate TimeSeries object for given province
def get_province_timeseries(df, province):
    series = TimeSeries.from_times_and_values(df.index, 
                                              df[province],
                                              fill_missing_dates=True,
                                              freq="W-MON")
    series = fill_missing_values(series)
    series = series.astype(np.float32)  
    return series

#Return a multivariate TimeSeries object for all provinces in dataframe
def get_provinces_timeseries(df, provinces):
    df = df.reset_index(names=['date'])
    series = TimeSeries.from_dataframe(df, 
                                       time_col='date', 
                                       value_cols=provinces,
                                       fill_missing_dates=True, 
                                       freq="W-MON")
    series = fill_missing_values(series)
    series = series.astype(np.float32)  
    return series


#Calculate Weighted-Interval Score of given prediction
def calculate_wis(truth, pred, alpha_vals):
    wis_vals = 0
    if hasattr(truth, "__len__"):
        wis_vals = np.zeros(len(truth))
    for alpha in alpha_vals:
        low_vals = pred.quantile_df(np.round(alpha/2,3)).clip(lower=0).iloc[:,0].values
        high_vals = pred.quantile_df(np.round(1-alpha/2,3)).clip(lower=0).iloc[:,0].values
        int_width = high_vals - low_vals
        too_low = (truth < low_vals).astype(int)
        too_high = (truth > high_vals).astype(int)
        too_low_penalty = (2/alpha)*too_low*(low_vals - truth)
        too_high_penalty = (2/alpha)*too_high*(truth - high_vals)
        wis_vals += (alpha/2)*(int_width + too_low_penalty + too_high_penalty)
    K = len(alpha_vals)-1
    wis_vals = wis_vals/(K+0.5)
    return wis_vals


def get_scaler(model_desc):
    if(model_desc=='ExponentialSmoothing'):
        scaler = Scaler(MinMaxScaler(feature_range=(1, 2)))
    else:
        scaler = Scaler()
    return scaler


#trains model by fitting timeseries upto pred_time 
#and predicts number of weeks_to_predict from pred_time
def fit_and_predict(series, model, model_desc, pred_date, weeks_to_predict, num_samples):

    start_time = time.time()
    train, _ = series.split_before(pred_date)
    transformer = get_scaler(model_desc)
    train_transformed = transformer.fit_transform(train)
    model = model.untrained_model()
    try:
        model.fit(train_transformed) 
        pred = model.predict(weeks_to_predict, num_samples=num_samples, verbose=False, show_warnings=False)
        pred = transformer.inverse_transform(pred)
    except (ValueError, TypeError) as err:
        warnings.warn(
            ("Unable to fit model {}. Throws error {}").format(model_desc, err))
    
    print_runtime = True
    if(print_runtime):
        runtime = (time.time() - start_time)
        print("date: {} --- {} seconds ---".format(pred_date, int(runtime)))

    return (pred.all_values())

#runs an expanding window backtesting procedure for given model 
#training and generating forecasts for each province separately
def run_backtests_local(df, provinces, model, model_desc, forecast_horizons, 
                         pred_dates, num_samples, results_dir):

    pred_dict = {}
    max_horizon = max(forecast_horizons)
    for province in provinces:
        print("-----------province={}-----------".format(province))
        series = get_province_timeseries(df, province)
        pred_all = []            
        for pred_date in pred_dates:
            pred = fit_and_predict(series, model, model_desc, pred_date, max_horizon, num_samples)
            pred_all.append(pred)
        for horizon in forecast_horizons:
            date_vals = pred_dates+pd.Timedelta(weeks=horizon)
            pred_vals = np.array([pred[horizon-1,:,:] for pred in pred_all])
            pred = TimeSeries.from_times_and_values(times=date_vals,values=pred_vals)
            pred = pred.map(lambda x: np.clip(x,0,np.inf)) 
            pred_dict[province + "_" +model_desc +"_" + str(horizon)] = pred
    
    results_fname = "{}/pred_results_{}.pkl".format(results_dir,model_desc)
    with open(results_fname, 'wb') as f:
        pickle.dump(pred_dict, f)              
    return pred_dict


#runs an expanding window backtesting procedure for given model 
#training and generating forecasts for all provinces together 
#(for models supporting global forecasting)
def run_backtests_global(df, provinces, model, model_desc, forecast_horizons, 
                         pred_dates, num_samples, results_dir):

    pred_dict = {}
    max_horizon = max(forecast_horizons)
    series = get_provinces_timeseries(df, provinces)
    pred_all = []            
    for pred_date in pred_dates:
        pred = fit_and_predict(series, model, model_desc, pred_date, max_horizon, num_samples)
        pred_all.append(pred)
    for horizon in forecast_horizons:
        date_vals = pred_dates+pd.Timedelta(weeks=horizon)
        for pind, province in enumerate(provinces):
            pred_vals = np.array([pred[horizon-1,pind,:] for pred in pred_all])
            pred_vals = np.expand_dims(pred_vals,1)
            pred = TimeSeries.from_times_and_values(times=date_vals,values=pred_vals)
            pred = pred.map(lambda x: np.clip(x,0,np.inf)) 
            pred_dict[province + "_" +model_desc +"_" + str(horizon)] = pred
    
    results_fname = "{}/pred_results_{}.pkl".format(results_dir,model_desc)
    with open(results_fname, 'wb') as f:
        pickle.dump(pred_dict, f)              
    return pred_dict


def plot_model_fit(province, series, pred, model, pred_date, weeks_to_predict, alpha_vals, output_dir):
            
    train, validation = series.split_before(pred_date)
    validation = validation.slice_n_points_after(validation.time_index[0],weeks_to_predict)  

    rmse_val = rmse(validation.slice_intersect(pred), pred)
    mape_val = mape(validation.slice_intersect(pred), pred)
    WIS = np.nan
    if(pred.is_probabilistic):
        truth = np.concatenate(validation.values())
        WIS = np.mean(calculate_wis(truth, pred, alpha_vals))

    plt.figure()
    train.plot(label="Train", alpha=1)
    validation.plot(label="Observed", alpha=1)
    pred.plot(label="Predict 95%", low_quantile=0.025, high_quantile=0.975, alpha = 0.25)
    pred.plot(label="Predict 50%", low_quantile=0.25, high_quantile=0.75, alpha = 0.5)
    pred.plot(label="Predict", low_quantile=None, high_quantile=None, alpha = 0.5, linewidth=1)
    plt.ylim(bottom=0)
    plt.suptitle(province, size=20, horizontalalignment='center')
    plt.title("Model: " + model + " | " +
                    "rmse=" + np.round(rmse_val,2).astype(str) +" | " +
                    "mape=" + np.round(mape_val,2).astype(str) + "% | " +
                    "WIS=" + np.round(WIS,2).astype(str), 
                    color="#404040", size=10,
                    verticalalignment="bottom")
    plt.savefig(output_dir +"/" + province + "_" + model + ".png", format="png", bbox_inches="tight", dpi=300)
    plt.show()


def calculate_and_plot_pred_fit(pred, series, province, model, horizon, df_metrics, alpha_vals, seasons, ax, color):
    
    obs = series[pred.time_index]
    rmse_val = rmse(obs, pred)
    mape_val = mape(obs, pred)
    smape_val = smape(obs, pred)

    wis_val = np.nan
    if(pred.is_probabilistic):
        truth = np.concatenate(obs.values())
        wis_vals = calculate_wis(truth, pred, alpha_vals)
        wis_val = np.mean(wis_vals)
    season = 'all'
    df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'rmse', rmse_val]
    df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'mape', mape_val]
    df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'smape', smape_val]
    df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'wis', wis_val]

    for season in seasons:
        time = obs.time_index[(obs.time_index>=seasons[season][0]) & (obs.time_index<=seasons[season][1])]
        obs_year = obs[time]
        pred_year = pred[time]
        rmse_val_year = rmse(obs_year, pred_year)
        mape_val_year = mape(obs_year, pred_year)
        smape_val_year = smape(obs_year, pred_year)

        wis_val_year = np.nan
        if(pred.is_probabilistic):
            truth = np.concatenate(obs_year.values())
            wis_val_year = np.mean(calculate_wis(truth, pred_year, alpha_vals))
        df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'RMSE', rmse_val_year]
        df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'MAPE', mape_val_year]
        df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'SMAPE', smape_val_year]
        df_metrics.loc[len(df_metrics.index)] = [province, model, horizon, season, 'WIS', wis_val_year]

    if(ax is not None):
        pred.plot(lw=2, label='{}, RMSE={:.2f}, MAPE={:.2f}%, SMAPE={:.2f}%, WIS={:.2f}'.format(
            model, rmse_val, mape_val, smape_val, wis_val),
                low_quantile=None, high_quantile=None,ax=ax,color=color,alpha=1)
        pred_low = pred.quantile_df(0.25).iloc[:,0].values
        pred_high = pred.quantile_df(0.75).iloc[:,0].values
        ax.fill_between(pred.time_index, pred_low, pred_high, color=color, alpha=0.5)
        pred_low = pred.quantile_df(0.025).iloc[:,0].values
        pred_high = pred.quantile_df(0.975).iloc[:,0].values
        ax.fill_between(pred.time_index, pred_low, pred_high, color=color, alpha=0.3)
        series.plot(label='_nolegend_',ax=ax, color='black')
        ax.legend(loc='upper center', fontsize=16)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=16)
        ax.set_xlabel("")
        ax.set_ylabel('weekly incidence rate per 100,000', fontsize=12)
        ax.set_title("")

    return df_metrics, wis_vals
