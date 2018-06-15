# CNN-blocking
Tool for optimize CNN blocking

usage: run_optimizer.py [-h] [-s SCHEDULE] [-v]
                        {basic,mem_explore, dataflow_explore} arch network

positional arguments:
  
  {basic,mem_explore, dataflow_explore}   optimizer type

  arch                  architecture specification

  network               network specification

optional arguments:

  -h, --help            show this help message and exit

  -s SCHEDULE, --schedule SCHEDULE restriction of the schedule space
  this is optional but restricting the schedule space will accelerate the scipt significantly

  -v, --verbose         vebosity


# Examples
## To optimize loop blocking.
Dataflow: Eyeriss

Memory Architecture: 3 level

Network: AlexNet Conv2 Batch16

```
python ./tools/run_optimizer.py -v -s ./examples/schedule/eyeriss_alex_conv2.json basic ./examples/arch/3_level_mem_baseline_asic.json ./examples/network/alex_conv2_batch16.json 
```

Dataflow: TPU

Memory Architecture: 3 level

Network: AlexNet Conv2 Batch16

```
python ./tools/run_optimizer.py -v -s ./examples/schedule/tpu.json basic ./examples/arch/3_level_mem_baseline_asic.json ./examples/network/alex_conv2_batch16.json
```

## To optimize memory capacity.
Dataflow: Eyeriss

Memory Architecture: 3 level

Network: AlexNet Conv2 Batch16

```
python ./tools/run_optimizer.py -v -s ./examples/schedule/eyeriss_alex_conv2.json mem_explore ./examples/arch/3_level_mem_explore.json ./examples/network/alex_conv2_batch16.json
```

## To explore dataflow.
Dataflow: All

Memory Architecture: Eyeriss

Network: AlexNet Conv2 Batch16

```
python ./tools/run_optimizer.py -v dataflow_explore ./examples/arch/3_level_mem_baseline_asic.json ./examples/network/alex_conv2_batch16.json
```

or:

```
python ./tools/run_optimizer.py -v -n user_defined_pickle_filename dataflow_explore ./examples/arch/3_level_mem_baseline_asic.json ./examples/network/alex_conv3_batch16.json
```

# Tutorials
## Understanding the scheduling results.
Lets say you are running a simple FC network on a simple asic.

Network:
    "input_fmap_channel":64, #64 input neurons
    
    "output_fmap_channel":48, #48 output neurons
    
    "batch_size":32
    
    #number of parameters: 64x48=3072
    
ASIC:
    "mem_levels": 2,
    
    "capacity":[256, 256000],
    
    "parallel_count":[64, 1]  # 64xPE array
    

```
python ./tools/run_optimizer.py -v ./examples/arch/fc_simple_arch.json ./examples/network/64x48fc_simple.json
```

You will get results like this:

```
best energy:  164480.0
cost for each level:  [103680.0, 896.0, 59904]
best schedule:  [[('IC', 16, 4), ('OC', 6, 8), None, None, None, None, None], [('ON', 32, 1), None, None, None, None, None, None]] ([[4], [5]], None
```

What does this mean?
Lets look at the scheduling results.

[('IC', 16, 4), ('OC', 6, 8), None, None, None, None, None]

The first part of the scheduling results tells the mapping schedule of the 1st memory hierarchy (PE array).
IC and OC will be mapped to the PE array.

![blocking](https://image.ibb.co/i8wHUd/1.png)

[‘IC’, 16, 4]

This tells that the input channel will be: 
*Blocked* with 16 IC unit each.
*Partitioned* through 4 row or column PE

[‘OC’, 6, 8]

This tells that the output channel will be: 
*Blocked* with 6 OC unit each.
*Partitioned* through 8 row or column PE

![mapping](https://image.ibb.co/cgNrpd/2.png)

Since the blocked IC and OC has a combination of 32 patterns, 
Each of the patterns will be mapped to a PE, utilizing 32PEs.


Now, the latter part in scheduling:

[('ON', 32, 1), None, None, None, None, None, None]] 

This tells the mapping schedule of the 2nd memory hierarchy (Buffer)
ON, or the batched input is executed as an outer loop.

To wrap up, as a schedule given

[[('IC', 16, 4), ('OC', 6, 8), None, None, None, None, None], [('ON', 32, 1), None, None, None, None, None, None]]

You can imagine a loopnest like this:

![mapping](https://image.ibb.co/diwsUd/image.jpg)

The weights are reused through ON loops, getting better efficiency!



