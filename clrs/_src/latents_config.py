import os
import numpy as np
from absl import logging

class LatentsConfig:
    def __init__(self):
        self.save_latents_flag = False
        self.filepath = None

    def set_latents_filepath(self, latents_path, algo_index, algorithms, seed, processor_type):
        task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
        latents_dir = os.path.join(latents_path, 'latents', task_type, f'{processor_type}_{seed}')
        os.makedirs(latents_dir, exist_ok=True)
        latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
        self.filepath = latents_filepath
    
    def save_latents(self, latents):
        latent_arrays = {key: np.array(latents[key]) for key in latents.keys()}
        np.savez(self.filepath, **latent_arrays)
        logging.info('Latents saved to %s', self.filepath)

latents_config = LatentsConfig()