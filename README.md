# food_freshness_detector :

This project uses Deep Learning to detect whether fruits are fresh or spoiled from images. It is built with PyTorch and uses a ResNet18 model as the backbone.

The project includes training and inference scripts, and allows interactive terminal-based predictions.

# Installation :

1.Clone the repository : 
    Initially, we need to clone the repository tp our local storage to access it, via the command :    
        `git clone https://github.com/Shyamkumar-M/food_freshness_detector.git`

    And, move into the directory.
        `cd food_freshness_detector`

2.Create and activate virtual environment :
    Create a virtual environment, it must be created only once
        `python -m venv venv`

    Activate it, this must be done at every new session
    
        `.\venv\Scripts\Activate.ps1`

3.Install required packages :
    Run
        `pip install torch torchvision pillow scikit-learn`

#Usage
1. Train the model