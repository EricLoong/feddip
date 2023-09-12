# FedDIP
## How to use?
1. All functions (procedures) used on clients are stored in **'client.py'**, including different training functions for different algorithms. 
2. Functions for the server to execure are stored in **‘fl_functions’**. 
3. Utilization tools are stored in **'functions_new.py'**. 
4. **'exp_args'** include all arguments for experiments. 
5. **'fl_server'** stores the FedDIP algorithm. If no 'prune' and 'sparse', it reduces to Fedavg.

## How to run experiments? (Example: FedDIP and FedDST; Definition of arguments refers to exp_args.py)
**Notice**: Use os.chdir() to change the directory to the folder of your project or use absolute path to run the experiments.
### 1. Run FedDIP on CIFAR10 with AlexNet
FedDIP: -python --epochs 1000 --num_users 50 --local_ep 5 --amount_sparsity 0.95 --model alexnet --dataset cifar10 --frac 0.1 --parallel 0 --num_workers 2 --reconfig_interval 5 --init_sparsity 0.5 --sparse --prune
### 2. Run FedDST on CIFAR100 with ResNet18
FedDST: -python feddst.py --feddst --epochs 1000 --num_users 50 --local_ep 5 --amount_sparsity 0.90 --model resnet18 --dataset cifar100 --frac 0.1 --parallel 0 --num_workers 2 --reconfig_interval 20 

### 3. Change differnt PRIFIX.py files to run different baseline algorithms according to their names. 

## Acknowledgements
Some code is modified based on the following repositories: 
1. https://github.com/rong-dai/DisPFL
2. https://github.com/jiangyuang/PruneFL
3. https://github.com/namhoonlee/snip-public