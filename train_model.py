import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, LeakyReLU, BatchNormalization, Conv1DTranspose, Input, Concatenate, MultiHeadAttention, Add, Activation, Multiply, MaxPooling1D)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from datetime import datetime

import time

from tensorflow.keras import mixed_precision

from tqdm import tqdm

from py_ml_tools.noise import get_noise
from dataclasses import dataclass

import matplotlib.pyplot as plt

import os

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

def save_diffusion_plots(output_data, labels, batch_idx, num_steps, N, file_name='../noisenet_data/diffusion_plots.png'):
    """
    Takes the output of the diffusion process and saves N evenly spaced subplots between step 0 and the max step.

    :param output_data: The output of the diffusion_process function.
    :param batch_idx: The index of the batch to use for creating the plots.
    :param N: The number of subplots to create.
    :param file_name: The name of the file to save the plot as (default: 'diffusion_plots.png').
    """
    
    steps_to_plot = np.linspace(0, num_steps - 1, N, dtype=int)

    # Create a new figure with a custom height based on the number of subplots
    plt.figure(figsize=(12, 3 * N))

    # Create a GridSpec with N rows and 1 column
    gs = gridspec.GridSpec(N, 2, height_ratios=[1]*N, hspace=0.5)

    for idx, step in enumerate(steps_to_plot):
        # Create a subplot using the GridSpec
        ax = plt.subplot(gs[idx*2])

        # Plot the specified step of the diffusion process for the specified batch
        ax.plot(tf.squeeze(output_data[step]).numpy())

        # Set title and labels
        ax.set_title(f'Step {step + 1}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        
        ax = plt.subplot(gs[idx*2 + 1])

        # Plot the specified step of the diffusion process for the specified batch
        ax.plot(tf.squeeze(labels[step]).numpy())

        # Set title and labels
        ax.set_title(f'Step Label')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')

    # Save the plot to a file
    plt.savefig(file_name)

    # Close the figure
    plt.close()

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

def process_generator_output_function(noise_samples, gps_times, num_steps):
    processed_noise_samples, conditions = tf.py_function(process_generator_output, [noise_samples, gps_times], [tf.float32, tf.float32])
    
    noisy_data_seq = diffusion_process(processed_noise_samples, num_steps)
    conditions = tf.repeat(conditions, num_steps, axis=0)
    targets = tf.repeat(processed_noise_samples, num_steps, axis=0)
    
    # Generate a shuffled index tensor
    batch_size = tf.shape(noisy_data_seq)[0]
    shuffled_indices = tf.argsort(tf.random.uniform((batch_size,), dtype=tf.float32), axis=-1)

    # Shuffle the tensors using the shuffled index tensor
    shuffled_noisy_data_seq = tf.gather(noisy_data_seq, shuffled_indices)
    shuffled_conditions = tf.gather(conditions, shuffled_indices)
    shuffled_targets = tf.gather(targets, shuffled_indices)
    
    return (noisy_data_seq, conditions), targets

def encoder_block(inputs, num_filters, kernel_size, attention_heads):
    x = Conv1D(num_filters, kernel_size, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = MultiHeadAttention(attention_heads, num_filters // attention_heads)(x, x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    skip = x

    x = MaxPooling1D(2, padding='same')(x)

    return x, skip


def middle_block(inputs, num_filters, kernel_size, attention_heads):
    x = Conv1D(num_filters, kernel_size, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = MultiHeadAttention(attention_heads, num_filters // attention_heads)(x, x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    return x


def decoder_block(inputs, skip, num_filters, kernel_size, attention_heads):
    x = Conv1DTranspose(num_filters, kernel_size, strides=2, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Concatenate()([x, skip])

    x = Conv1D(num_filters, kernel_size, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = MultiHeadAttention(attention_heads, num_filters // attention_heads)(x, x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    return x

def positional_encoding(inputs, pos_embedding_dim):
    input_shape = tf.shape(inputs)
    batch_size, seq_length, _ = input_shape[0], input_shape[1], input_shape[2]

    position_indices = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    div_terms = tf.exp(-tf.math.log(10000.0) * (tf.range(0, pos_embedding_dim, 2, dtype=tf.float32) / pos_embedding_dim))
    pos_encodings = position_indices * div_terms

    sin_encodings = tf.sin(pos_encodings)
    cos_encodings = tf.cos(pos_encodings)

    pos_encodings = tf.stack([sin_encodings, cos_encodings], axis=2)
    pos_encodings = tf.reshape(pos_encodings, (1, seq_length, pos_embedding_dim))

    return tf.cast(pos_encodings, dtype=tf.float32)

def create_conditional_attention_unet(input_shape, num_conditions, num_filters=32, kernel_size=3, attention_heads=8, pos_embedding_dim=16):
    audio_input = Input(shape=input_shape)
    conditions_input = Input(shape=(num_conditions,))

    # Add initial 1D convolution for feature extraction
    x = Conv1D(num_filters, kernel_size, padding='same')(audio_input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    # Add conditions
    repeated_conditions = tf.repeat(tf.expand_dims(conditions_input, axis=1), input_shape[0], axis=1)
    #x = Concatenate(axis=-1)([x, repeated_conditions])

    # Add positional encoding
    pos_encodings = positional_encoding(x, pos_embedding_dim)
    x_with_pos = Concatenate(axis=-1)([x, pos_encodings])

    # Encoder
    encoder_outputs = []
    x = x_with_pos
    for i in range(2):
        x, skip = encoder_block(x, num_filters * (2 ** i), kernel_size, attention_heads)
        encoder_outputs.append(skip)

    # Middle
    x = middle_block(x, num_filters * 4, kernel_size, attention_heads)

    # Decoder
    for i in reversed(range(2)):
        x = decoder_block(x, encoder_outputs[i], num_filters * (2 ** i), kernel_size, attention_heads)

    # Output
    output = Conv1DTranspose(input_shape[-1], kernel_size, padding='same', activation='linear')(x)

    model = Model(inputs=[audio_input, conditions_input], outputs=output)

    return model
    
def diffusion_process(processed_noise_samples, num_steps):
    expanded_input = tf.expand_dims(processed_noise_samples, axis=1) 
    gaussian_noise = tf.random.normal(tf.shape(expanded_input), dtype=tf.float32)
    
    # Generate the linear noise_stddev_seq tensor with the correct shape
    linear_seq = tf.linspace(0.0, 1.0, num_steps)
    noise_stddev_seq = tf.reshape(linear_seq, [1, -1, 1, 1])
    
    noise_shape = tf.shape(processed_noise_samples)
    
    # Tile the noise_stddev_seq tensor along the first axis to match the first dimension of processed_noise_samples
    noise_stddev_seq_expanded = tf.tile(noise_stddev_seq, [noise_shape[0], 1, noise_shape[1], noise_shape[2]])
    noisy_data_seq = expanded_input * (1.0 - noise_stddev_seq_expanded) + gaussian_noise * noise_stddev_seq_expanded
    
    data_seq_shape = tf.shape(noisy_data_seq)
    
    new_first_dim = (data_seq_shape[0] * data_seq_shape[1])
    return tf.reshape(noisy_data_seq, (new_first_dim, data_seq_shape[2], data_seq_shape[3]))

def train_conditional_denoising_model(model, generator, max_num_examples, num_steps, batch_size, output_path):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, int(sample_rate_hertz * example_duration_seconds), 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
        )
    )
    dataset = dataset.map(
        lambda noise_samples, gps_times: process_generator_output_function(noise_samples, gps_times, num_steps),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Calculate the total number of steps per epoch
    steps_per_epoch = batch_size*num_steps

    checkpoint_callback = ModelCheckpoint(
        filepath=f"{output_path}/noisenet_{epoch:03d}.h5",
        save_freq='epoch',
        monitor="loss",
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    model.fit(
        dataset,
        batch_size = batch_size,
        epochs = int(max_num_examples / steps_per_epoch*batch_size),
        steps_per_epoch = steps_per_epoch,
        callbacks = [checkpoint_callback],
        verbose = 1
    )
        
def generate_conditional_audio(model, initial_noise, conditions, noise_stddev_seq):
    current_data = initial_noise
    for t, noise_stddev in enumerate(reversed(noise_stddev_seq)):
        prediction = model.predict([current_data, conditions])
        current_data = prediction + np.random.normal(0, noise_stddev, current_data.shape)
    return current_data

O3 = (
    "O3",
    datetime(2019, 4, 1, 0, 0, 0),
    datetime(2020, 3, 27, 0, 0, 0)
)

observing_run_data = (O3,)
observing_runs = {}

for run in observing_run_data:
    observing_runs[run[0]] = ObservingRun(run[0], run[1], run[2])             

start = observing_runs["O3"].start_gps_time
stop  = observing_runs["O3"].end_gps_time

minimum_length = 1.0
channel = "DCS-CALIB_STRAIN_CLEAN_C01"
frame_type = "HOFT_C01"
state_flag = "DCS-ANALYSIS_READY_C01:1"

batch_size = 8
num_steps = 16

max_num_examples = 1E6

sample_rate_hertz = 1024.0
example_duration_seconds = 1.0

def return_gen():
    return get_noise(
        start = start,
        stop = stop,
        ifo = "L1",
        sample_rate_hertz = sample_rate_hertz,
        channel = channel,
        frame_type = frame_type,
        state_flag = state_flag,
        example_duration_seconds = example_duration_seconds,
        max_num_examples = max_num_examples,
        num_examples_per_batch = batch_size,
        order = "shortest"
    )

if __name__ == "__main__":
    
    strategy = setup_CUDA(True, "1,2,3,4,5,6,7")
    print("CUDA setup complete.")
    
    with strategy.scope():
        epochs = 10
        
        output_path = "../noisenet_outputs"

        input_shape = (int(sample_rate_hertz * example_duration_seconds), 1)
        num_conditions = 9

        model = create_conditional_attention_unet(input_shape, num_conditions)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss='mean_squared_error')
        
        train_conditional_denoising_model(
            model, 
            return_gen, 
            max_num_examples, 
            num_steps, 
            batch_size, 
            output_path
        )