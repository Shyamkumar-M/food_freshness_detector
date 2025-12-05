# food_freshness_detector :

This project uses Deep Learning to detect whether fruits are fresh or spoiled from images. It is built with PyTorch and uses a ResNet18 model as the backbone.

The project includes training and inference scripts, and allows interactive terminal-based predictions.

# Project Structure :
food_freshness_detector/
│
├── data/                     
│   ├── train/
│   │   ├── fresh/
│   │   └── spoiled/
│   ├── val/
│   └── test/
│
├── models/
│   ├── baseline.py            
│   └── __init__.py
│
├── saved_models/              
│   └── resnet18_freshness.pth
│
├── scripts/
│   ├── train.py               
│   ├── inference.py           
│   └── gui_inference.py       
└── venv/                      

# Installation :

1.Clone the repository : 
    Initially, we need to clone the repository tp our local storage to access it, via the commands :    
        git clone https://github.com/Shyamkumar-M/food_freshness_detector.git