
import subprocess
import sys
def install_packages(packages):
    """
    Import helper to import required packages
    """    
    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"Installing {package}...")
            
            try :
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} has been installed.")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}.")



    # List of packages to install

    # Call the function to install the packages
    import subprocess

    import pandas as pd
    import numpy as np

    import tensorflow as tf
    import matplotlib.pyplot as plt
