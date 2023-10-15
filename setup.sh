#!/bin/bash

# create virtual environment
python3 -m venv venv

# activate virtual environment
source venv/bin/activate

# install packages from requirements.txt
pip install -r requirements.txt
