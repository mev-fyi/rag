import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Append the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

