#!/bin/bash

set -e
set -x

# set up virtualenv
python3 -m venv '.venv'
source '.venv/bin/activate'

# install torch and torchvision
wget 'https://gist.githubusercontent.com/codeandfire/5b98dac9a5453e765f1c212625b118b2/raw/bbaa53a6e5feca3680ca026c09d908d854791b76/install_pytorch.sh'
/bin/bash ./install_pytorch.sh 'v'
rm 'install_pytorch.sh'

# install other packages
pip3 install -r 'requirements.txt'
pip3 install 'lime'

# collect an image dataset
git clone 'https://github.com/codeandfire/imagenet-scraper.git'
cd 'imagenet-scraper'
# goldfish, house finch, bulbul, balloon, bathing cap, analog clock, digital clock, muzzle, geyser
/bin/bash ./scraper.sh -n '150' 'n01443537' 'n01532829' 'n01560419' 'n02782093' 'n02807133' 'n02708093' 'n03196217' 'n03803284' 'n09288635'
cd '..'
mkdir 'dataset'
mv imagenet-scraper/n*/ 'dataset'
# this list of 1000 classes will come in handy
cp 'imagenet-scraper/wnids_1000.txt' '.'
rm -r -f 'imagenet-scraper'

# download a pretrained AlexNet model
python3 -c "import torch; model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True);"

set +x
