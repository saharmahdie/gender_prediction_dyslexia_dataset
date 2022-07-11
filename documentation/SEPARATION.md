# Seperation in SMIDA
[List of Documentation Files](menu.md)

The ```eye_movement_classification``` class provides the oportunity to select different separation algorithm.
At the Moment only IVT is implemented

## Optional Input

With ```mark_nan=True``` all NaN will be listed in sample overview.

## IVT (trajectory_split.py)

Split data by velocity threshold.

1. find possible fixations by given threshold
2. combine fixations shorter given minimal duration

Return a list of Saccades and Fixation by Tupel [Start,End].

## Sliding Window?