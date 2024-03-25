from jax.experimental import host_callback
import numpy as np
from absl import logging
import os

class LatentsConfig:
    def __init__(self):
        self.save_latents_flag = False
        self.filepath = None

    def save_latents_to_file(self, filepath, latents):

        def save_callback(latents):
            np_latents = {k: np.asarray(v) for k, v in latents.items()}
            np.savez(filepath, **np_latents)
            logging.info(f"Latents saved to {filepath}")

        host_callback.call(save_callback, latents)
    
    def set_latents_filepath(self, algo_index, algorithms, seed, processor_type):
        # Infer task type, processor type, and seed from FLAGS
        task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
        latents_dir = os.path.join('/weights/latents', task_type, f'{processor_type}_{seed}')
        os.makedirs(latents_dir, exist_ok=True)
        latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
        self.filepath = latents_filepath

latents_config = LatentsConfig()