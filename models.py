import numpy as np
import itertools
import torch
from darts.models import ExponentialSmoothing, LightGBMModel, NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.utils import ModelMode, SeasonalityMode

import darts.utils.likelihood_models as Likelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

quantiles = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.050)),[0.975,0.99])

# Pytorch early stopping rules
my_stopper = EarlyStopping(
    monitor="train_loss",  # Over which values we are optimizing
    patience=10,           # After how many iterations if the loss doesn't improve then it stops optimizing
    min_delta=0.001,       # Round-off error to consider that it didn't improved
    mode="min"
)

cuda_available = torch.cuda.is_available()
print("GPU available: {}".format(cuda_available))
if(cuda_available):
    # device = 'gpu'
    pl_trainer_kwargs1={"callbacks": [my_stopper], "accelerator": 'gpu', "devices": -1}
else:
    # device = 'cpu'
    pl_trainer_kwargs1 = {"callbacks": [my_stopper], "accelerator": "cpu"}


#Return the defined lists of local and global forecasting models
def get_forecasting_models(data_type): 

    # List of local models
    model_list_local = {
        "ExponentialSmoothing": ExponentialSmoothing(
            seasonal_periods=52, 
            trend=ModelMode.ADDITIVE, 
            damped=True,
            seasonal=SeasonalityMode.MULTIPLICATIVE),
    }
    # List of global models
    model_list_global = {
        "LightGBM_Global": LightGBMModel(
                lags=[-1,-2,-53], #
                lags_past_covariates=2, 
                lags_future_covariates=[2,2], 
                output_chunk_length=1,
                likelihood="quantile",
                quantiles=quantiles,
                # add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]},
                                        'transformer': Scaler() },
                show_warnings=False,
                verbose=-1
            ),
        "NHiTS_Global": NHiTSModel(
                input_chunk_length=26 if(data_type=='malaria') else 52,
                output_chunk_length=1,
                likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
                n_epochs=200,
                num_stacks=5,
                num_blocks=2,
                dropout=0.1,
                optimizer_kwargs={'lr': 1e-3}, 
                pl_trainer_kwargs=pl_trainer_kwargs1,
                add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
                show_warnings=False,
            ),
    }
    return (model_list_local, model_list_global)