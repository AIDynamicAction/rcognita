#!/usr/bin/env python

import argparse

# required arg

parser = argparse.ArgumentParser()
   
parser.add_argument('--t0', required=True)

args = parser.parse_args()

print(f'Hello {args.t0}')