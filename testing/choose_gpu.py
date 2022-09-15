import py3nvml
free_gpus = py3nvml.get_free_gpus()
print(free_gpus)
if True not in free_gpus:
    print('No free gpus found')