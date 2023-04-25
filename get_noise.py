from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf

from dataclasses import dataclass
from get_background import get_background_examples

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

def main():
    
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
        
    background_noise_iterator = get_background_examples(
        start = start,
        stop = stop,
        ifo = "L1",
        sample_rate_hertz = 1024.0,
        channel = channel,
        frame_type = frame_type,
        state_flag = state_flag,
        example_duration_seconds = 1.0,
        max_num_examples = 32,
        num_examples_per_batch = 32
    )
    
    for i, noise_chunk in enumerate(background_noise_iterator):
        
        print(noise_chunk)
        print(i*32)
        
if __name__ == "__main__":
    
    setup_CUDA(True, "5")
    main()

    
   