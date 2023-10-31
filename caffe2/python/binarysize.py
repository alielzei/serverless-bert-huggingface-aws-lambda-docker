"""A tool to inspect the binary size of a built binary file.

This script prints out a tree of symbols and their corresponding sizes, using
Linux's nm functionality.

Usage:

    python binary_size.py --             --target=/path/to/your/target/binary             [--nm_command=/path/to/your/custom/nm]             [--max_depth=10] [--min_size=1024]             [--color] 
To assist visualization, pass in '--color' to make the symbols color coded to
green, assuming that you have a xterm connection that supports color.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import subprocess
import sys


class Trie(object):
    """A simple class that represents a Trie."""
    
    def __init__(self, name):
        """Initializes a Trie object."""
        self.name = name
        self.size = 0
        self.dictionary = {}


def GetSymbolTrie(target, nm_command, max_depth):
    """Gets a symbol trie with the passed in target.

    Args:
            target: the target binary to inspect.
            nm_command: the command to run nm.
            max_depth: the maximum depth to create the trie.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.binarysize.GetSymbolTrie', 'GetSymbolTrie(target, nm_command, max_depth)', {'subprocess': subprocess, 'sys': sys, 'Trie': Trie, 'target': target, 'nm_command': nm_command, 'max_depth': max_depth}, 1)

def MaybeAddColor(s, color):
    """Wrap the input string to the xterm green color, if color is set.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.binarysize.MaybeAddColor', 'MaybeAddColor(s, color)', {'s': s, 'color': color}, 1)

def ReadableSize(num):
    """Get a human-readable size."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.binarysize.ReadableSize', 'ReadableSize(num)', {'num': num}, 1)

def PrintTrie(trie, prefix, max_depth, min_size, color):
    """Prints the symbol trie in a readable manner.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.binarysize.PrintTrie', 'PrintTrie(trie, prefix, max_depth, min_size, color)', {'MaybeAddColor': MaybeAddColor, 'ReadableSize': ReadableSize, 'PrintTrie': PrintTrie, 'trie': trie, 'prefix': prefix, 'max_depth': max_depth, 'min_size': min_size, 'color': color}, 0)

def main(argv):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.binarysize.main', 'main(argv)', {'sys': sys, 'argparse': argparse, 'GetSymbolTrie': GetSymbolTrie, 'PrintTrie': PrintTrie, 'argv': argv}, 0)
if __name__ == '__main__':
    main(sys.argv[1:])

