#!/bin/bash
(
  export PYTHONPATH=$(realpath ..)

  python learn.py && 
  python normalize.py && 
  python histograms.py && 
  python checker.py
)
