## Resource-Adaptive Successive Doubling for Hyperparameter Optimization with Large Datasets on High-Performance Computing Systems

This repository contains the code of the Resource-Adaptive Successive Doubling Algorithm (RASDA), a scheduler for performing efficient hyperparameter optimization within Ray Tune by reallocating GPU resources. 
The main part is the resource allocation functionality, which can be found in _res_allocated.py_ It requires the following input:
- base_cpus, base_gpus: The amount of computing resource allocated to each worker (ideally base_gpus = 1)
- num_workers: The amount of initial parallel workers allocated to each trial
- scaling_factor: The scaling factor by which the number of workers is multiplied at each milestone
- min_t, max_t: The minimum and maximum amount of iterations to train per trial

The scheduler should be used in addition to the ASHA scheduler, with the similar values for min_t, max_t and reduction_factor=scaling_factor. See _adaptive_ray.py_ for an example of usage on the ImageNet dataset.
 
