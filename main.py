import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, LeakyReLU, BatchNormalization, Conv1DTranspose, Input, Concatenate)
from tensorflow.keras.models import Model
from datetime import datetime

from tqdm import tqdm

from get_background import get_background_examples
from dataclasses import dataclass

import os


def setup_CUDA(verbose, device_num):
		
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
		
	gpus =  tf.config.list_logical_devices('GPU')
	strategy = tf.distribute.MirroredStrategy(gpus)

	physical_devices = tf.config.list_physical_devices('GPU')
	
	for device in physical_devices:	

		try:
			tf.config.experimental.set_memory_growth(device, True)
		except:
			# Invalid device or cannot modify virtual devices once initialized.
			pass
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	if verbose:
		tf.config.list_physical_devices("GPU")
		
	return strategy

@dataclass
class ObservingRun:
    def __init__(self, name: str, start_date_time: datetime, end_date_time: datetime):
        self.name = name
        self.start_date_time = start_date_time
        self.end_date_time = end_date_time
        self.start_gps_time = self._to_gps_time(start_date_time)
        self.end_gps_time = self._to_gps_time(end_date_time)
        
    def _to_gps_time(self, date_time: datetime) -> float:
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        time_diff = date_time - gps_epoch
        leap_seconds = 18 # current number of leap seconds as of 2021 (change if needed)
        total_seconds = time_diff.total_seconds() - leap_seconds
        return total_seconds

def get_conditions_from_gps_time(gps_time_scalar_tensor):
    gps_time = gps_time_scalar_tensor.numpy()
    dt = datetime.utcfromtimestamp(int(gps_time))

    # Time of year
    day_of_year = dt.timetuple().tm_yday
    days_in_year = 366 if dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0) else 365
    time_of_year = (day_of_year - 1) / (days_in_year - 1)

    # Day of week
    day_of_week = np.zeros(7)
    day_of_week[dt.weekday()] = 1

    # Time of day
    time_of_day = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400

    conditions = np.concatenate([np.array([time_of_year, time_of_day]), day_of_week])
    
    return conditions.astype(np.float32)

def get_conditions_from_gps_time_wrapper(gps_time):
    return tf.py_function(get_conditions_from_gps_time, [gps_time], tf.float32)

@tf.function
def process_generator_output(noise_samples, gps_times):
    batch_size = gps_times.shape[0]

    # Initialize the conditions tensor with the correct shape
    conditions = tf.TensorArray(tf.float32, size=batch_size, dynamic_size=False)

    for i in tf.range(batch_size):
        conditions = conditions.write(i, get_conditions_from_gps_time_wrapper(gps_times[i]))

    return noise_samples, conditions.stack()

def create_conditional_1d_denoising_model(input_shape, num_conditions, num_filters=32, kernel_size=3):
    audio_input = Input(shape=input_shape)
    conditions_input = Input(shape=(num_conditions,))
    
    x = Conv1D(num_filters, kernel_size, padding='same')(audio_input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    # Repeat the conditions and concatenate with the feature maps
    repeated_conditions = tf.repeat(tf.expand_dims(conditions_input, axis=1), input_shape[0], axis=1)
    x = Concatenate(axis=-1)([x, repeated_conditions])
    
    x = Conv1D(num_filters * 2, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(num_filters * 2, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(num_filters, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    output = Conv1DTranspose(input_shape[-1], kernel_size, padding='same', activation='linear')(x)
    
    model = Model(inputs=[audio_input, conditions_input], outputs=output)
    return model

def diffusion_process(data, noise_stddev_seq):
    noisy_data_seq = []
    current_data = data[..., np.newaxis]  # Add the channel dimension
    for noise_stddev in noise_stddev_seq:
        current_data = current_data + np.random.normal(0, noise_stddev, current_data.shape)
        noisy_data_seq.append(current_data)
    return np.stack(noisy_data_seq, axis=1)


def train_conditional_denoising_model(model, generator, num_batches, noise_stddev_seq, batch_size):
    for batch_num, (noise_samples, gps_times) in tqdm(enumerate(generator)):
        print(f"Training on batch {batch_num + 1}/{num_batches}")
        processed_noise_samples, conditions = process_generator_output(noise_samples, gps_times)
        
        noisy_data_seq = diffusion_process(processed_noise_samples, noise_stddev_seq)

        for t in reversed(range(noisy_data_seq.shape[1])):
            model.fit([noisy_data_seq[:, t], conditions], processed_noise_samples, epochs=1, batch_size=batch_size, verbose=2)
        
def generate_conditional_audio(model, initial_noise, conditions, noise_stddev_seq):
    current_data = initial_noise
    for t, noise_stddev in enumerate(reversed(noise_stddev_seq)):
        prediction = model.predict([current_data, conditions])
        current_data = prediction + np.random.normal(0, noise_stddev, current_data.shape)
    return current_data

ifos = ("H1", "L1", "V1")

O1 = (
    "O1",
    datetime(2015, 9, 12, 0, 0, 0),
    datetime(2016, 1, 19, 0, 0, 0)
)

O2 = (
    "O2",
    datetime(2016, 11, 30, 0, 0, 0),
    datetime(2017, 8, 25, 0, 0, 0)
)

O3 = (
    "O3",
    datetime(2019, 4, 1, 0, 0, 0),
    datetime(2020, 3, 27, 0, 0, 0)
)

observing_run_data = (O1, O2, O3)
observing_runs = {}

for run in observing_run_data:
    observing_runs[run[0]] = ObservingRun(run[0], run[1], run[2])             

start = observing_runs["O3"].start_gps_time
stop  = observing_runs["O3"].end_gps_time

minimum_length = 1.0
channel = "DCS-CALIB_STRAIN_CLEAN_C01"
frame_type = "HOFT_C01"
state_flag = "DCS-ANALYSIS_READY_C01:1"

batch_size = 32

max_num_examples = 1E6
num_batches = int(max_num_examples / batch_size)

sample_rate_hertz = 1024.0
example_duration_seconds = 1.0

background_noise_iterator = get_background_examples(
    start = start,
    stop = stop,
    ifo = "L1",
    sample_rate_hertz = sample_rate_hertz,
    channel = channel,
    frame_type = frame_type,
    state_flag = state_flag,
    example_duration_seconds = example_duration_seconds,
    max_num_examples = max_num_examples,
    num_examples_per_batch = batch_size
)
    
noise_stddev_seq = np.linspace(0.01, 0.5, 50)  # Noise standard deviation sequence for diffusion process
epochs = 10

input_shape = (int(sample_rate_hertz * example_duration_seconds), 1)
num_conditions = 9

model = create_conditional_1d_denoising_model(input_shape, num_conditions)

model.compile(optimizer='adam', loss='mean_squared_error')
train_conditional_denoising_model(model, background_noise_iterator, num_batches, noise_stddev_seq, batch_size)