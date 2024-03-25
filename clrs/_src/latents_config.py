from jax.experimental import host_callback
import numpy as np
from absl import logging
import os
import jax

class LatentsConfig:
    def __init__(self):
        self.save_latents_flag = False
        self.filepath = None
        self.latents_list = []
        self.num_batches = 0

    def save_latents_to_file(self, filepath, latents):
        def save_callback(latents):
            self.latents_list.append(latents)
            if len(self.latents_list) == self.num_batches:
                accumulated_latents = jax.tree_util.tree_multimap(
                    lambda *args: np.concatenate(args, axis=0), *self.latents_list)
                np_latents = {k: np.asarray(v) for k, v in accumulated_latents.items()}
                np.savez(filepath, **np_latents)
                logging.info(f"Latents saved to {filepath}")
                self.latents_list.clear()

        host_callback.call(save_callback, latents)

    def set_latents_filepath(self, latents_path, algo_index, algorithms, seed, processor_type, num_batches):
        # Infer task type, processor type, and seed from FLAGS
        task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
        latents_dir = os.path.join(latents_path, 'latents', task_type, f'{processor_type}_{seed}')
        os.makedirs(latents_dir, exist_ok=True)
        latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
        self.filepath = latents_filepath
        self.num_batches = num_batches

latents_config = LatentsConfig()