from jax.experimental import io_callback
import numpy as np
from absl import logging
import os
import jax
import jax.numpy as jnp

class LatentsConfig:
    def __init__(self):
        self.save_latents_flag = False
        self.filepath = None
        self.latents_list = []
        self.current_batch = 0
    
    def print_latents(self, latents):
        print(latents)
    
    def accumulate_callback(self, latents):

        def accumulate_latents(latents):
            self.latents_list.append(latents)
            print("Number of batches processed:", len(self.latents_list))

        io_callback(accumulate_latents, None, latents)

    def save_latents(self):
        accumulated_latents = jax.tree_util.tree_map(
                lambda *args: jnp.stack(args, axis=0), *self.latents_list)
        print("Number of batches reached.")
        print("Doing jax tree map")
        np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
        print("Finished jax tree map")
        print("Saving latents to file.")
        np.savez(self.filepath, **np_latents)
        logging.info(f"Latents saved to {self.filepath}")
        self.latents_list.clear()


    #         self.current_batch += 1

    #         if self.current_batch == self.num_batches:
    #             accumulated_latents = jax.tree_util.tree_map(
    #                 lambda *args: jnp.stack(args, axis=0), *self.latents_list)
    #             print("Number of batches reached.")
    #             print("Doing jax tree map")
    #             np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
    #             print("Finished jax tree map")
    #             print("Saving latents to file.")
    #             np.savez(self.filepath, **np_latents)
    #             logging.info(f"Latents saved to {self.filepath}")
    #             self.latents_list.clear()
    #             self.current_batch = 0

    # def accumulate_latents(self, latents):
    #     self.latents_list.append(latents)
    #     print("Number of batches processed:", len(self.latents_list))
    #     self.current_batch += 1

    #     # Don't need callback, save to NumPy file directly
    #     if self.current_batch == self.num_batches:
    #         print("Number of batches reached.")
    #         print("Doing jax tree map")
    #         accumulated_latents = jax.tree_util.tree_map(
    #             lambda *args: jnp.stack(args, axis=0), *self.latents_list)
    #         print("Finished jax tree map")
    #         print("Converting to NumPy")
    #         np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
    #         print("Saving latents to file.")
    #         np.savez(self.filepath, **np_latents)
    #         logging.info(f"Latents saved to {self.filepath}")
    #         self.latents_list.clear()
    #         self.current_batch = 0

    def set_latents_filepath(self, latents_path, algo_index, algorithms, seed, processor_type, num_batches):
        task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
        latents_dir = os.path.join(latents_path, 'latents', task_type, f'{processor_type}_{seed}')
        os.makedirs(latents_dir, exist_ok=True)
        latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
        self.filepath = latents_filepath
        self.num_batches = num_batches
        

latents_config = LatentsConfig()

    # EXTRACT LATENTS PER BATCH
    # ACCUMULATE LATENTS FROM ALL BATCHES
    # SAVE LATENTS TO FILE
        
    # def extract_latents(self, latents):

    # def accumulate_latents(self, latents):

    # def save_latents(self, latents):


    # def accumulate_and_save_latents(self, latents):

    #     if self.save_latents_flag:
    #             self.latents_list.append(latents)
    #             print("Number of latents appended to list:", len(self.latents_list))
    #             self.current_batch += 1
    #             print("Number of batches processed:", self.current_batch)
        
        # def accumulate_and_save_latents_callback(latents):
        #     if self.save_latents_flag:
        #         self.latents_list.append(latents)
        #         print("Number of latents appended to list:", len(self.latents_list))
        #         self.current_batch += 1
        #         print("Number of batches processed:", self.current_batch)

        #         if self.current_batch == self.num_batches:
        #             accumulated_latents = jax.tree_util.tree_map(
        #                 lambda *args: jnp.stack(args, axis=0), *self.latents_list)
        #             print("Number of batches reached.")
        #             print("Doing jax tree map")
        #             np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
        #             print("Finished jax tree map")
        #             print("Saving latents to file.")
        #             np.savez(self.filepath, **np_latents)
        #             logging.info(f"Latents saved to {self.filepath}")
        #             self.latents_list.clear()
        #             self.current_batch = 0

        # io_callback(accumulate_and_save_latents_callback, None, latents)

    # def accumulate_and_save_latents(self, latents):
        
    #     if self.save_latents_flag:
    #         self.latents_list.append(latents)
    #         print("Number of latents appended to list:", len(self.latents_list))
    #         self.current_batch += 1
    #         print("Number of batches processed:", self.current_batch)

    #         if self.current_batch == self.num_batches:
    #             accumulated_latents = jax.tree_util.tree_map(
    #                 lambda *args: jnp.stack(args, axis=0), *self.latents_list)
    #             io_callback(lambda _: self.save_callback(accumulated_latents), None)

    # def accumulate_and_save_latents(self, latents):
    #     if self.save_latents_flag:
    #         def accumulate_callback(latents):
    #             self.latents_list.append(latents)
    #             print("Number of latents appended to list:", len(self.latents_list))
    #             self.current_batch += 1
    #             print("Number of batches processed:", self.current_batch)

    #             if self.current_batch == self.num_batches:
    #                 accumulated_latents = jax.tree_util.tree_map(
    #                     lambda *args: jnp.stack(args, axis=0), *self.latents_list)
    #                 self.save_callback(accumulated_latents)

    #         io_callback(accumulate_callback, None, latents)

    # def accumulate_and_save_latents(self, latents):
    #     if self.save_latents_flag:
    #         self.latents_list.append(latents)
    #         print("Number of latents appended to list:", len(self.latents_list))
    #         self.current_batch += 1
    #         print("Number of batches processed:", self.current_batch)

    #         if self.current_batch == self.num_batches:
    #             accumulated_latents = jax.tree_util.tree_map(
    #                 lambda *args: jnp.stack(args, axis=0), *self.latents_list)

    #             io_callback(self.save_callback, accumulated_latents)

    # def save_latents(self, accumulated_latents):
    #     print("Number of batches reached.")
    #     print("Doing jax tree map")
    #     np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
    #     print("Finished jax tree map")
    #     print("Saving latents to file.")
    #     np.savez(self.filepath, **np_latents)
    #     logging.info(f"Latents saved to {self.filepath}")
    #     self.latents_list.clear()
    #     self.current_batch = 0
        
#     def save_latents(self, latents):
#         # print("Number of batches reached.")
#         # print("Doing jax tree map")
#         # np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
#         # print("Finished jax tree map")
#         print("Saving latents to file.")
#         np.savez(self.filepath, **latents)
#         logging.info(f"Latents saved to {self.filepath}")
#         # self.latents_list.clear()
#         # self.current_batch = 0

#     def set_latents_filepath(self, latents_path, algo_index, algorithms, seed, processor_type, num_batches):
#         task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
#         latents_dir = os.path.join(latents_path, 'latents', task_type, f'{processor_type}_{seed}')
#         os.makedirs(latents_dir, exist_ok=True)
#         latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
#         self.filepath = latents_filepath
#         self.num_batches = num_batches

# latents_config = LatentsConfig()

# class LatentsConfig:
#     def __init__(self):
#         self.save_latents_flag = False
#         self.filepath = None
#         self.latents_list = []
#         self.current_batch = 0

#     def accumulate_and_save_latents(self, latents):
#         if self.save_latents_flag:
#             self.latents_list.append(latents)
#             print("Latents appended to list")
#             self.current_batch += 1

#             if self.current_batch == self.num_batches:
#                 accumulated_latents = jax.tree_util.tree_map(
#                     lambda *args: jnp.stack(args, axis=0), *self.latents_list)

#                 def save_callback(accumulated_latents):
#                     np_latents = jax.tree_util.tree_map(np.asarray, accumulated_latents)
#                     np.savez(self.filepath, **np_latents)
#                     logging.info(f"Latents saved to {self.filepath}")
#                     self.latents_list.clear()
#                     self.current_batch = 0

#                 host_callback.call(save_callback, accumulated_latents)

#     def set_latents_filepath(self, latents_path, algo_index, algorithms, seed, processor_type, num_batches):
#         task_type = 'single_task' if len(algorithms) == 1 else 'multi_task'
#         latents_dir = os.path.join(latents_path, 'latents', task_type, f'{processor_type}_{seed}')
#         os.makedirs(latents_dir, exist_ok=True)
#         latents_filepath = os.path.join(latents_dir, f'{algorithms[algo_index]}.npz')
#         self.filepath = latents_filepath
#         self.num_batches = num_batches

# latents_config = LatentsConfig()