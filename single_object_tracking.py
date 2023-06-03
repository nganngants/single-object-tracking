

from eval import evalTLP, evalTLPAtrr
from demo import demo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-eval', required=True, choices=['demo', 'tiny-tlp', 'tlp', 'tlp-attr'], type=str)
parser.add_argument('-show', help='show tracking process when evaluating dataset or not', default=False, type=bool)
args = parser.parse_args()

if __name__ == '__main__':
  data = args.eval
  dir_TLP = '../TLP'
  dir_tinyTLP = './TinyTLP'
  dir_TLPAttr = './TLPAttr'
  if data == 'demo':
    demo()
  elif data == 'tiny-tlp':
    evalTLP(dir_tinyTLP, args.show)
  elif data == 'tlp-attr':
    evalTLPAtrr(dir_TLPAttr, args.show)
  else:
    evalTLP(dir_TLP, args.show)
