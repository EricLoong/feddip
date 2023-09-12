import argparse
import os
os.umask(0o002)

parser = argparse.ArgumentParser()
# Federated Training arguments
parser.add_argument('--epochs', metavar='EP', type=int, required=True,
                    help='number of iteration of federated training')
parser.add_argument('--num_users', type=int, required=True,
                    help="number of clients: K")
parser.add_argument('--frac', type=float, default=0.1,
                    help='the fraction of clients: C')
parser.add_argument('--local_ep', type=int, required=True,
                    help="the number of local epochs: E")
parser.add_argument('--local_batchsize', type=int, default=128,
                    help="local batch size: B")
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')

# Weight Approximation arguments
parser.add_argument('--df', type=int, default=1,
                    help='degree of freedom of polynomial approximation')
parser.add_argument('--group', type=int, default=3,
                    help='how many number of elements in one group')
parser.add_argument('--amount_err', type=float, default=0.1, help='The amount of errors kept')
parser.add_argument('--apprx', type=int, default=1,
                    help='Decide whether conduct weight approximation, 1 means yes')

# Prune and sparse arguments
parser.add_argument("--erk_power_scale", type=float, default=1)
parser.add_argument('--amount_sparsity', type=float, required=True,
                    help='Percentage of zeros accounts for each pruned model')
parser.add_argument('--sparse', action='store_true',
                    help='Decide whether conduct sparse learning, 1 means True')
parser.add_argument('--prune', action='store_true',
                    help='Decide whether conduct prune, 1 means True')
parser.add_argument('--lambda_shrink', type=float, default=1e-03,
                    help='Penalty affect the shrinkage rate')
parser.add_argument('--pnorm', choices=[1, 2, float('inf')],type=int, default=2,
                    help='Penalty affect the shrinkage rate')
parser.add_argument('--init_sparsity', type=float, default=0.5,
                    help='Initial sparsity for pruning')
# Dataset and Models
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset used for experiments')
parser.add_argument('--model', type=str, default='alexnet',
                    help='Model used for experiments')
parser.add_argument('--prt', type=int, default=0,
                    help='Adopt pretrained model or not')
parser.add_argument('--tfstp', type=int, default=0,
                    help='Train starting from stop points')
parser.add_argument('--model_filename', type=str, default='None',
                    help='The filename of the stored model')
parser.add_argument('--parallel', type=int, default=0,
                    help='Adopt multiple GPUs or not')
parser.add_argument('--num_workers', type=int, default=2,
                    help='The number of workers for DataLoaders')
parser.add_argument('--partition_method', type=str, default='iid', choices=['iid','pat', 'dir'],
                    help='Partition method: "pat" (default iid) or "dir"')


# PruneFL parameters
parser.add_argument('--init_rounds', type=int, default='50',
                    help='The initial rounds to do the initial pruning')
parser.add_argument('--reconfig_interval', type=int, default='20',
                    help='The rounds/interval to do the reconfiguration')
parser.add_argument("--prunefl", action='store_true', help='whether to conduct prunefl')


# SNIP
parser.add_argument("--snip", action='store_true', help='whether to conduct SNIP')

# PruneTrain
parser.add_argument('--prunetrain_power', type=float, default='0.01',
                    help='The rounds/interval to do the reconfiguration')
# FedProx
parser.add_argument("--prox", action='store_true', help='whether to conduct FedProx')
parser.add_argument("--mu", type=float, default=0.01, help='regularization parameter for proximal term')

# FedDST
parser.add_argument('--anneal_factor', type=float, default='0.5',
                    help='The fraction to remove and regrow in FedDST')
parser.add_argument("--feddst", action='store_true', help='whether to conduct FedDST')


################################################All Arguments Involved##################################################
args = parser.parse_args()





