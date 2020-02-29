from __future__ import print_function
import argparse
import torch
import os, shutil
from dataset_utils import load_blackbox
import numpy as np
import pickle

# Training settings
parser = argparse.ArgumentParser(description='Auxiliary Information construction')
# parser.add_argument('--batch-size', type=int, default=128, metavar='')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
# parser.add_argument('--epochs', type=int, default=100, metavar='')
# parser.add_argument('--lr', type=float, default=0.01, metavar='')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
# parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--truncation', type=int, default=530)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--auxiliary', type=int, default=100)

def get_inversion_data(model, label, quantity, nz=530):
    # add Gaussian noise
    Gaussian_noise = np.random.normal(0.0, 0.1, (quantity, nz))
    for i in range(quantity):
        Gaussian_noise[i][label] += 1.0

    input_vector = torch.from_numpy(Gaussian_noise)
    input_vector = input_vector.float()
    results = model(input_vector).cpu()
    auxiliary_data = []
    for i in range(quantity):
        data = (results[i].detach().numpy(), input_vector[i])
        auxiliary_data.append(data)
    return auxiliary_data

def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")
    os.makedirs('models/adversary/', exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    inversion = load_blackbox('inversion.pth', args=args, device=device)
    auxiliary = []
    for i in range(args.nz):
        auxiliary.append(get_inversion_data(inversion, i, args.auxiliary, args.nz))
    np.random.shuffle(auxiliary)
    pickle.dump(auxiliary, 'models/adversary/transferset.pickle')


if __name__ == '__main__':
    main()