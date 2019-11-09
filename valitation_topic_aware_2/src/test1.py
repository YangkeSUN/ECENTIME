import argparse
import json

import valide_util as util

def load_config(config_path):
    with util.smart_open(config_path) as config_file:
        return json.load(config_file)
#end def load_config

def main():
    parser = argparse.ArgumentParser(description='recommend films to groups of user')
    parser.add_argument('--config', nargs=1, metavar='FILE', type=str, required=False, default='config.json'
            , help='configuration file containing input file locations and default parameters (default=config.json)')
    parser.add_argument('--nb_k', nargs=1, metavar='K', type=float, required=False, help='number of links in group of user')
    parser.add_argument('--alpha', nargs=1, metavar='A', type=float, required=False, help='Social welfare coeficient (between 0.0 and 1.0)')
    parser.add_argument('--beta', nargs=1, metavar='B', type=float, required=False, help='faireness coeficient (between 0.0 and 1.0)')
    parser.add_argument('--m', nargs=1, metavar='M', type=float, required=False, help="number of users in a group")
    parser.add_argument('--x', nargs=1, metavar='X', type=float, required=False, help="minimum number of links from one user to other users in a group")
    args = parser.parse_args()
    
    config = load_config(args.config[0])
    if (args.alpha != None):
        config["alpha"] = args.alpha[0]
    if (args.beta != None):
        config["beta"] = args.beta[0]
    if (args.m != None):
        config["m"] = args.m[0]
    if (args.x != None):
        config["x"] = args.x[0]
    if (args.nb_k != None):
        config["nb_k"] = args.nb_k[0]
    print(config)

main()
