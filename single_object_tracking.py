

from eval import evalTinyTLP, evalTLPAtrr
from demo import demo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-eval', required=True, default='demo', choices=['tiny-tlp', 'demo', 'tlp-attr'], type=str)
parser.add_argument('-show', help='show tracking process on video or not', default=False, type=bool)
args = parser.parse_args()

if __name__ == '__main__':
  data = args.eval
  if data == 'demo':
    demo()
  if data == 'tiny-tlp':
    evalTinyTLP(args.show)
  if data == 'tlp-attr':
    evalTLPAtrr(args.show)
"""
  to create requirements.txt, run:
  py -m pipreqs.pipreqs .
"""