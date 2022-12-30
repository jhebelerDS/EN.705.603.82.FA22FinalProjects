#### Dependencies ####
Python 3.6.8 \
Pytorch 0.3.1.post3 \
Numpy 1.15.2 \
Fastrand from https://github.com/lemire/fastrand \
Gym 0.10.5 \
Mujoco-py v1.50.1.59

#### Install process:  #####
$conda create -n HIERL \
$conda install python==3.6.8 \
$conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch \
$conda install numpy==1.19.5 \
$pip install gym==0.15.6 \
$pip install -U kaleido
 
Setup mujoco-py: \
$git clone https://github.com/openai/mujoco-py.git \
$cd mujoco-py \
$python setup.py install \
Fastrand from https://github.com/lemire/fastrand 

Setup gym-go:
$git clone https://github.com/SeanTBlack/GymGo.git \
$cd GymGo \
$pip install -e .

#### To Run #### 
python run_erl.py -env $ENV_NAME$ 

#### ENVS TESTED #### 
gym-go

#### Algorithms Implemented #### 
Algorithm Name          Branch \
ERL                 |   go_noHER \
NS-ERL              |   go_NS 
