#!/bin/bash
(
  export PYTHONPATH=$(realpath ..)

  python normalize.py && 
  python histograms.py && 
  python checker.py
)
