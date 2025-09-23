import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.ocean_basins import get_region
from utils.data_loader import load_as_maps, load_for_mlp
from utils.plot_map import plot_map

def plot_shap_over_time(model, X, mask, feature_names=None, 
                              bg_per_time=64, max_samples_per_time=None,
                              batch_size=None, verbose=True, region_id='ARCTIC', type='TREND'):
    """
    Parameters
    ----------
    model : Keras model
    X_tnf : np.ndarray, shape (T, N, F)
        T time steps, N samples per step, F features.
    feature_names : list[str] or None
    bg_per_time : int
        How many samples per time step to use for the SHAP background (masker).
        We'll aggregate across time to form a background of size ~T*bg_per_time.
    max_samples_per_time : int or None
        If set, randomly subsample this many rows per time step for speed.
    batch_size : int or None
        Passed to SHAP prediction calls (helpful for big models).
    """

    region = get_region(X, region_id)
    mask_expanded = np.expand_dims(mask, axis=0)
    region_mask = np.squeeze(get_region(mask_expanded, region_id))
    filtered_region = region[:, region_mask, :]


    T, N, F = filtered_region.shape
    num_years = T // 12

    if type == 'TREND':
        filtered_region = filtered_region.reshape(num_years, 12, N, F).reshape(num_years, 12*N, F)
    else:
        filtered_region = filtered_region.reshape(num_years, 12, N, F).transpose(1, 0, 2, 3).reshape(12, num_years*N, F)

    T, N, F = filtered_region.shape
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(F)]

    # Build a small background set from all times (stratified by time)
    rng = np.random.default_rng(42)
    take = min(bg_per_time, N)
    bg_list = []
    for t in range(T):
        idx = rng.choice(N, size=take, replace=False)
        bg_list.append(filtered_region[t, idx])
    background = np.concatenate(bg_list, axis=0)  # (~T*bg_per_time, F)

    # Wrap predictor so SHAP can batch efficiently
    def f(X):
        return model.predict(X, batch_size=batch_size, verbose=0)

    # Create explainer once
    explainer = shap.Explainer(f, background, feature_names=feature_names)

    # Storage for mean |SHAP| per (time, feature)
    mean_abs_shap = np.zeros((T, F), dtype=float)

    print(T)

    for t in range(T):
        X_t = filtered_region[t]  # (N, F)

        if max_samples_per_time is not None and max_samples_per_time < len(X_t):
            idx = rng.choice(len(X_t), size=max_samples_per_time, replace=False)
            X_t_eval = X_t[idx]
        else:
            X_t_eval = X_t

        sv = explainer(X_t_eval)           # sv.values shape: (M, F)
        vals = np.abs(sv.values)           # mean absolute SHAP per feature
        mean_abs_shap[t] = vals.mean(axis=0)

        if verbose and (t % max(1, T//10) == 0):
            print(f"Processed time step {t+1}/{T}")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    title = ''
    xlabel = ''
    
    if type == 'TREND':
        xlabel = "Year"
        match region_id:
            case "ARCTIC":
                title = "Feature Importance Trend in the Arctic region"
            case "NORTH_ATLANTIC":
                title = "Feature Importance Trend in the North Atlantic"
            case "EQ_PACIFIC":
                title = "Feature Importance Trend in the quatorial Pacific"
            case "SOUTHERN_OCEAN":
                title = "Feature Importance Trend in the Southern Ocean"
    else:
        xlabel = "Month"
        match region_id:
            case "ARCTIC":
                title = "Seasonal Feature Importance in the Arctic region"
            case "NORTH_ATLANTIC":
                title = "Seasonal Feature Importance in the North Atlantic"
            case "EQ_PACIFIC":
                title = "Seasonal Feature Importance in the quatorial Pacific"
            case "SOUTHERN_OCEAN":
                title = "Seasonal Feature Importance in the Southern Ocean"
        
    # Plot 1
    plt.figure(figsize=(12, 6))
    for j, name in enumerate(feature_names[:6]):
        plt.plot(months, mean_abs_shap[:, j], label=name)

    plt.xlabel(xlabel)
    plt.ylabel("Mean |SHAP value|")
    plt.title(title)
    plt.legend(
    ncol=3, fontsize=9,
    bbox_to_anchor=(0.5, -0.15),
    loc='upper center'
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2
    plt.figure(figsize=(12, 6))
    for j, name in enumerate(feature_names[6:]):
        plt.plot(months, mean_abs_shap[:, 6+j], label=name)

    plt.xlabel(xlabel)
    plt.ylabel("Mean |SHAP value|")
    plt.title(title)
    plt.legend(
    ncol=3, fontsize=9,
    bbox_to_anchor=(0.5, -0.15),
    loc='upper center'
    )
    plt.grid(True)   
    plt.tight_layout()
    # plt.savefig(path+'/yearly_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')
    plt.show()

    return mean_abs_shap, feature_names

# plot mean predicted and true co2 flux over time per basin
def plot_timeseries_analysis(targets, predictions, region, path, start_year):
    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape

    yearly_target_mean = region_targets.mean(axis=(1, 2))
    yearly_pred_mean = region_pred.mean(axis=(1, 2))

    # monthly std across years (±1σ band). Use ddof=1 for sample std if you prefer.
    #targ_std = yearly_target_mean.std(axis=0, ddof=1)
    #pred_std = yearly_pred_mean.std(axis=0, ddof=1)

    months = np.arange(T)
    years = start_year + months / 12

    plt.figure(figsize=(10, 5))
    plt.plot(years, yearly_target_mean, label='Target', color='blue')
    plt.plot(years, yearly_pred_mean, label='Prediction', color='orange')

    # ±1σ shaded bands
    #plt.fill_between(years, yearly_target_mean - targ_std, yearly_target_mean + targ_std, alpha=0.2, color='blue', label='Target ±1σ')
    #plt.fill_between(years, yearly_pred_mean - pred_std, yearly_pred_mean + pred_std, alpha=0.2, color='orange', label='Prediction ±1σ')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Time")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/timeseries_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')


# do the same but just over one year, or summarize by month category
def plot_seasonal_analysis(targets, predictions, region,path):
    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape
    num_years = T // 12

    targets_reshaped = region_targets.reshape(num_years, 12, H, W)
    pred_reshaped = region_pred.reshape(num_years, 12, H, W)
    # yearly_pred_mean = pred_yearly.mean(axis=(0, 2, 3))
    # yearly_target_mean = targets_yearly.mean(axis=(0, 2, 3))

    # monthly means per year: (years, months)
    targ_mean_per_year_month = targets_reshaped.mean(axis=(2, 3))
    pred_mean_per_year_month = pred_reshaped.mean(axis=(2, 3))

    # grand monthly mean across years: (months,)
    targ_mean = targ_mean_per_year_month.mean(axis=0)
    pred_mean = pred_mean_per_year_month.mean(axis=0)

    # monthly std across years (±1σ band). Use ddof=1 for sample std if you prefer.
    targ_std = targ_mean_per_year_month.std(axis=0, ddof=1)
    pred_std = pred_mean_per_year_month.std(axis=0, ddof=1)

    x = np.arange(12)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 5))
    plt.plot(months, targ_mean, label='Target', color='blue')
    plt.plot(months, pred_mean, label='Prediction', color='orange')

    # ±1σ shaded bands
    plt.fill_between(x, targ_mean - targ_std, targ_mean + targ_std, alpha=0.2, color='blue', label='Target ±1σ')
    plt.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, alpha=0.2, color='orange', label='Prediction ±1σ')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Month")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/seasonal_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')



# plot mean and predicted co2 flux averaged over year and plot for multiple years
def plot_yearly_analysis(targets, predictions, region, path, start_year):
    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape
    num_years = T // 12

    targets_yearly = region_targets.reshape(num_years, 12, H, W)
    yearly_target_mean = targets_yearly.mean(axis=(1, 2, 3))

    pred_yearly = region_pred.reshape(num_years, 12, H, W)
    yearly_pred_mean = pred_yearly.mean(axis=(1, 2, 3))

    years = np.arange(start_year, start_year+num_years)  
    plt.figure(figsize=(10, 5))
    plt.plot(years, yearly_target_mean, label='Target', color='blue')
    plt.plot(years, yearly_pred_mean, label='Prediction', color='orange')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Year")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/yearly_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')


def complete_model_analysis_map(model_path, dataset_id):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test, Y_test = load_as_maps(start_year=2013, end_year=2018, datasets=[dataset_id])
    map_mask = X_test[0,:,:, 11] == 1

    scaler = load(model_path + '/scaler.pkl')
    n_samples, h, w, n_features = X_test.shape
    X_test_flat = X_test.reshape(-1,n_features)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test = X_test_scaled_flat.reshape(n_samples, h, w, n_features)

    # run model and evalutate
    predictions = model.predict(X_test)
    predictions = predictions.reshape(n_samples, h, w)
    pred_masked = predictions * map_mask

    pred_flat = predictions.reshape(-1)
    mask = X_test_flat[:, 10] == 1
    pred_flat_masked = pred_flat[mask]

    Y_test_flat = Y_test.reshape(-1)
    Y_test_flat_masked = Y_test_flat[mask]

    general_mse = mean_squared_error(pred_flat_masked, Y_test_flat_masked)
    general_mae = mean_absolute_error(pred_flat_masked, Y_test_flat_masked)
   

    with open(model_path + "/model_evaluation.txt", "a") as f:
        f.write("Evaluation " + dataset_id + ":\n")
        f.write(f"General MSE: {general_mse:.3f}\n")
        f.write(f"General MAE: {general_mae:.3f}\n\n")

        regions = ['ARCTIC', 'NORTH_ATLANTIC', 'EQ_PACIFIC', 'SOUTHERN_OCEAN']

        for region in regions:
            region_targets = get_region(Y_test, region).reshape(-1)
            region_pred = get_region(pred_masked, region).reshape(-1)

            mse = mean_squared_error(region_pred, region_targets)
            mae = mean_absolute_error(region_pred, region_targets)
            f.write(f"MSE {region}: {mse:.3f}\n")
            f.write(f"MAE {region}: {mae:.3f}\n")
        
        f.write("\n"+  "-" * 30 + "\n")


    # compute error maps
    absolute_error = np.abs(pred_masked - Y_test)
    mean_error = np.mean(absolute_error, axis=0)
    squared_error = (pred_masked - Y_test) ** 2
    rmse = np.sqrt(np.mean(squared_error, axis=0)) 

    # plot error maps
    plot_map(data=mean_error, folder_path=model_path, title='Mean absolute error co2 flux pre reconstruction 2013 - 2018', file_name='mae_' + dataset_id,vmin=0, vmax=1, cmap='summer')
    plot_map(data=rmse, folder_path=model_path, title='Root mean squared error co2 flux pre reconstruction 2013 - 2018', file_name='rmse_' + dataset_id,vmin=0, vmax=1, cmap='summer')

    start_year = 2013

    path = model_path + "/" + dataset_id
    if not os.path.exists(path):
        os.makedirs(path)

    plot_timeseries_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

    plot_seasonal_analysis(Y_test, pred_masked, "ARCTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "EQ_PACIFIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path)

    plot_yearly_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

def complete_model_analysis_mlp(model_path, dataset_id):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test_flat, Y_test_flat = load_for_mlp(start_year=2013, end_year=2018, datasets=[dataset_id])

    mask = X_test_flat[:, 10] == 1
    X_test_flat = np.delete(X_test_flat, [10], axis=1)

    scaler = load(model_path + '/scaler.pkl')
    n_samples, n_features = X_test_flat.shape
    X_test_scaled_flat = scaler.transform(X_test_flat)

    map_samples = int(n_samples/167/360)

    map_mask = mask.reshape(map_samples, 167, 360)
    Y_test = Y_test_flat.reshape(map_samples, 167, 360)

    # run model and evalutate
    pred_flat = model.predict(X_test_scaled_flat)
    pred_flat_masked = pred_flat[mask]

    Y_test_flat_masked = Y_test_flat[mask]

    general_mse = mean_squared_error(pred_flat_masked, Y_test_flat_masked)
    general_mae = mean_absolute_error(pred_flat_masked, Y_test_flat_masked)

    pred_masked = pred_flat.reshape(map_samples, 167, 360) * map_mask
   

    with open(model_path + "/model_evaluation.txt", "a") as f:
        f.write("Evaluation " + dataset_id + ":\n")
        f.write(f"General MSE: {general_mse:.3f}\n")
        f.write(f"General MAE: {general_mae:.3f}\n\n")

        regions = ['ARCTIC', 'NORTH_ATLANTIC', 'EQ_PACIFIC', 'SOUTHERN_OCEAN']

        for region in regions:
            region_targets = get_region(Y_test, region).reshape(-1)
            region_pred = get_region(pred_masked, region).reshape(-1)

            mse = mean_squared_error(region_pred, region_targets)
            mae = mean_absolute_error(region_pred, region_targets)
            f.write(f"MSE {region}: {mse:.3f}\n")
            f.write(f"MAE {region}: {mae:.3f}\n")
        
        f.write("\n"+  "-" * 30 + "\n")


    # compute error maps
    absolute_error = np.abs(pred_masked - Y_test)
    mean_error = np.mean(absolute_error, axis=0)
    squared_error = (pred_masked - Y_test) ** 2
    rmse = np.sqrt(np.mean(squared_error, axis=0)) 

    # plot error maps
    plot_map(data=mean_error, folder_path=model_path, title='Mean absolute error co2 flux pre reconstruction 2013 - 2018', file_name='mae_' + dataset_id,vmin=0, vmax=1, cmap='summer')
    plot_map(data=rmse, folder_path=model_path, title='Root mean squared error co2 flux pre reconstruction 2013 - 2018', file_name='rmse_' + dataset_id,vmin=0, vmax=1, cmap='summer')

    start_year = 2013

    path = model_path + "/" + dataset_id
    if not os.path.exists(path):
        os.makedirs(path)

    plot_timeseries_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

    plot_seasonal_analysis(Y_test, pred_masked, "ARCTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "EQ_PACIFIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path)

    plot_yearly_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)