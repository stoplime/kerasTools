# List out all the possible builtin losses and metrics
from keras import metrics, losses, optimizers
from enum import Enum, auto

loss = {}

class losses_enum (Enum):
    mean_squared_error = auto()
    mean_absolute_error = auto()
    mean_absolute_percentage_error = auto()
    mean_squared_logarithmic_error = auto()
    squared_hinge = auto()
    hinge = auto()
    categorical_hinge = auto()
    logcosh = auto()
    categorical_crossentropy = auto()
    sparse_categorical_crossentropy = auto()
    binary_crossentropy = auto()
    kullback_leibler_divergence = auto()
    poisson = auto()
    cosine_proximity = auto()

loss[losses_enum.mean_squared_error] = losses.mean_squared_error
loss[losses_enum.mean_absolute_error] = losses.mean_absolute_error
loss[losses_enum.mean_absolute_percentage_error] = losses.mean_absolute_percentage_error
loss[losses_enum.mean_squared_logarithmic_error] = losses.mean_squared_logarithmic_error
loss[losses_enum.squared_hinge] = losses.squared_hinge
loss[losses_enum.hinge] = losses.hinge
loss[losses_enum.categorical_hinge] = losses.categorical_hinge
loss[losses_enum.logcosh] = losses.logcosh
loss[losses_enum.categorical_crossentropy] = losses.categorical_crossentropy
loss[losses_enum.sparse_categorical_crossentropy] = losses.sparse_categorical_crossentropy
loss[losses_enum.binary_crossentropy] = losses.binary_crossentropy
loss[losses_enum.kullback_leibler_divergence] = losses.kullback_leibler_divergence
loss[losses_enum.poisson] = losses.poisson
loss[losses_enum.cosine_proximity] = losses.cosine_proximity

metric = {}

class metrics_enum (Enum):
    mean_squared_error = auto()
    mean_absolute_error = auto()
    mean_absolute_percentage_error = auto()
    mean_squared_logarithmic_error = auto()
    squared_hinge = auto()
    hinge = auto()
    logcosh = auto()
    categorical_crossentropy = auto()
    sparse_categorical_crossentropy = auto()
    binary_crossentropy = auto()
    kullback_leibler_divergence = auto()
    poisson = auto()
    cosine_proximity = auto()
    binary_accuracy = auto()
    categorical_accuracy = auto()
    sparse_categorical_accuracy = auto()
    top_k_categorical_accuracy = auto()
    sparse_top_k_categorical_accuracy = auto()

metric[metrics_enum.mean_squared_error] = metrics.mean_squared_error
metric[metrics_enum.mean_absolute_error] = metrics.mean_absolute_error
metric[metrics_enum.mean_absolute_percentage_error] = metrics.mean_absolute_percentage_error
metric[metrics_enum.mean_squared_logarithmic_error] = metrics.mean_squared_logarithmic_error
metric[metrics_enum.squared_hinge] = metrics.squared_hinge
metric[metrics_enum.hinge] = metrics.hinge
metric[metrics_enum.logcosh] = metrics.logcosh
metric[metrics_enum.categorical_crossentropy] = metrics.categorical_crossentropy
metric[metrics_enum.sparse_categorical_crossentropy] = metrics.sparse_categorical_crossentropy
metric[metrics_enum.binary_crossentropy] = metrics.binary_crossentropy
metric[metrics_enum.kullback_leibler_divergence] = metrics.kullback_leibler_divergence
metric[metrics_enum.poisson] = metrics.poisson
metric[metrics_enum.cosine_proximity] = metrics.cosine_proximity
metric[metrics_enum.binary_accuracy] = metrics.binary_accuracy
metric[metrics_enum.categorical_accuracy] = metrics.categorical_accuracy
metric[metrics_enum.sparse_categorical_accuracy] = metrics.sparse_categorical_accuracy
metric[metrics_enum.top_k_categorical_accuracy] = metrics.top_k_categorical_accuracy
metric[metrics_enum.sparse_top_k_categorical_accuracy] = metrics.sparse_top_k_categorical_accuracy

optimizer = {}

class optimizers_enum (Enum):
    SGD = auto()
    RMSprop = auto()
    Adagrad = auto()
    Adadelta = auto()
    Adam = auto()
    Adamax = auto()
    Nadam = auto()
    TFOptimizer = auto()

optimizer[optimizers_enum.SGD] = optimizers.SGD
optimizer[optimizers_enum.RMSprop] = optimizers.RMSprop
optimizer[optimizers_enum.Adagrad] = optimizers.Adagrad
optimizer[optimizers_enum.Adadelta] = optimizers.Adadelta
optimizer[optimizers_enum.Adam] = optimizers.Adam
optimizer[optimizers_enum.Adamax] = optimizers.Adamax
optimizer[optimizers_enum.Nadam] = optimizers.Nadam
optimizer[optimizers_enum.TFOptimizer] = optimizers.TFOptimizer