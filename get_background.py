from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import List

import numpy as np
import tensorflow as tf
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries

@dataclass
class NoiseChunk:
    noise : tf.Tensor
    t0    : float
    dt    : float
    
    def random_subsection(self, num_subsection_elements: int, num_examples_per_batch: int):      
        # Check if the input tensor is 1D
        assert len(self.noise.shape) == 1, "Input tensor must be 1D"

        # Get the length of the input tensor
        N = self.noise.shape[0]

        # Ensure X is smaller or equal to N
        assert num_subsection_elements <= N, "X must be smaller or equal to the length of the tensor"

        # Generate a random starting index for each element in the batch
        maxval = tf.cast(N - num_subsection_elements + 1, dtype=tf.int32)
        random_starts = tf.random.uniform((num_examples_per_batch,), minval=0, maxval=maxval, dtype=tf.int32)

        # Extract the subsections of the tensor for the entire batch
        indices = tf.expand_dims(tf.range(num_subsection_elements, dtype=tf.int32), 0) + random_starts[:, tf.newaxis]
        batch_subtensors = tf.gather(self.noise, indices)

        # Calculate the t0 for each captured subsection
        t0_subsections = self.t0 + tf.cast(random_starts, dtype=tf.float32) * self.dt

        return tf.expand_dims(batch_subtensors, axis=-1), t0_subsections
    
def timeseries_to_noise_chunk(timeseries : TimeSeries, scale_factor : float):
    data = tf.convert_to_tensor(timeseries.value * scale_factor, dtype = tf.float32)
    return NoiseChunk(data, timeseries.t0.value, timeseries.dt.value)

def get_background_examples(
    start: float,
    stop: float,
    ifo: str,
    sample_rate_hertz: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    saturation: float = 1.0,
    example_duration_seconds: float = 1.0,
    max_num_examples: float = 0.0,
    num_examples_per_batch: int = 1,
    scale_factor: float = 1.0,
    order: str = "random",
):
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}"],
        start,
        stop,
    )

    intersection = segments[f"{ifo}:{state_flag}"].active.copy()

    valid_segments = []

    for seg_start, seg_stop in intersection:
        if (seg_stop - seg_start) >= example_duration_seconds*num_examples_per_batch:
            valid_segments.append((seg_start, seg_stop))

    if not valid_segments:
        raise ValueError("No segments of minimum length, not producing background")

    if order == "random":
        shuffle(valid_segments)
    elif order == "shortest_first":
        valid_segments = sorted(valid_segments, key=lambda duration: duration[1] - duration[0])

    def get_new_segment_data(segment_start, segment_end):
        files = find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=segment_start,
            gpsend=segment_end,
            urltype="file",
        )

        data = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=segment_start, end=segment_end
        )
        data = data.resample(sample_rate_hertz)
        data = data.whiten(4, 2)

        if np.isnan(data).any():
            raise ValueError(f"The background for ifo {ifo} contains NaN values")

        return timeseries_to_noise_chunk(data, scale_factor)

    current_segment_index = 0
    current_example_index = 0

    while current_segment_index < len(valid_segments):
        current_segment_start, current_segment_end = valid_segments[current_segment_index]
        current_max_sample_count = int((current_segment_end - current_segment_start) / saturation)

        try:
            print(f"Loading segments of duration {current_segment_end - current_segment_start}")
            current_segment_data = get_new_segment_data(current_segment_start, current_segment_end)
            print("Complete!")

            for _ in range(current_max_sample_count):
                num_subsection_elements = int(example_duration_seconds * sample_rate_hertz)
                batch_noise_data = current_segment_data.random_subsection(num_subsection_elements, num_examples_per_batch)
                
                current_example_index += num_examples_per_batch;
                
                if (max_num_examples > 0) and (current_example_index > max_num_examples):
                    return
                                    
                yield batch_noise_data

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}, {str(e)}")

        current_segment_index += 1
