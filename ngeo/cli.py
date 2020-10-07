"""
ngeo is not an NGO

Usage:
  ngeo train --csv-file=<csv-file> --output-model=<output-model>
  ngeo test --csv-file=<csv-file> --model=<model>
  ngeo predict <name>

Options:
  --csv-file=<csv-file>     CSV file with name and class information
"""

from docopt import docopt


def main():
    args = docopt(__doc__)

    raise NotImplementedError()
