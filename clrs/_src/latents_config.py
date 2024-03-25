from jax.experimental import host_callback
import numpy as np

class LatentsConfig:
    def __init__(self):
        self.save_latents_flag = False
        self.filepath = None

    def save_latents_to_file(self, filepath, latents):

        def save_callback(latents):
            np_latents = {k: np.asarray(v) for k, v in latents.items()}
            np.savez(filepath, **np_latents)

        host_callback.call(save_callback, latents)

latents_config = LatentsConfig()