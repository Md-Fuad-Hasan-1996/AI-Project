This framework of foundation models for autonomous driv-
ing uses ViNT and NoMaD pre-trained models (Checkpoints).
NoMaD uses goal masking diffusion policies navigation and
exploration which is an advanced version of ViNT model. Also
NoMaD is developed on top of ViNT architecture. Here is the
step-by-step procedure we followed to evaluate our real-world
datasets for autonomous driving using both ViNT and NoMaD
pre-trained checkpoints:
A. Pre-requisites
The codebase requires a workstation with Python 3.7+,
Ubuntu (tested on 18.04 and 20.04), and a GPU with CUDA
10+ installed. Although you can adapt it to operate with dif-
ferent virtual environment packages or a native configuration,
it also presumes access to Conda.
B. Project Setup
Run the following commands inside the root directory of
the project.In our case the default name of the root directory
is foundation_model
1) Set up the conda environment:
conda env create -f train
/train_environment.yaml
2) Activate the conda environment with the exact name
defined in train_environment.yaml:
conda activate vint_train
3) Install the vint_train packages:
pip install -e train/
4) Install the diffusion_policy package from the
following repo:
https://github.com/real-stanford
/diffusion_policy
Commands:
git clone git@github
.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
C. Data Processing
The datsets should be processed as the following structure:
< d a ta s e t _ na me >   
+−− < n a m e _ o f _ t r a j 1 >   
| +−− 0 . j p g  
| +−− 1 . j p g  
| +−− . . .  
| +−− T_1 . j p g  
| \ − − t r a j _ d a t a . p k l  
+−− < n a m e _ o f _ t r a j 2 >
| +−− 0 . j p g
| +−− 1 . j p g
| +−− . . .
| +−− T_2 . j p g
| \ − − t r a j _ d a t a . p k l
. . .
\ − − < name_of_trajN >
+−− 0 . j p g
+−− 1 . j p g
+−− . . .
+−− T_N . j p g
\ − − t r a j _ d a t a . p k l
1) Add the data set root directory path to the relevant
model’s config files i.e. vint.yaml for ViNT and no-
mad.yaml for NoMaD models.
2) Run the command inside the root directory of the
project:
python <path to the data_split.py file>
After running this command, the processed
data-split should the following structure
inside vint_release/train/vint_train
/data/data_splits/
< da ta se t _ na m e >
+−− t r a i n
| \ − − \ − − t e s t
\ − − t r a j _ n a m e s . t x t
t r a j _ n a m e s . t x t
So now the data is structured for both training and
evaluation.
D. Training with Pre-trained Checkpoints
1) For training set train: False inside the related
model config file: For example to train
with ViNT checkpoints, make this change to
/foundation_model/train/config/vint.yaml
2) Download the pre-trained models from this link: Pre-
trained models
3) Add:load_run:<project_name>/
<log_run_name> to your .yaml config file
in foundation_model/train/config/.
The *.pth of the file you are loading to
be saved in this file structure and renamed
to “latest”: foundation_model/train
/logs/<project_name>/<log_run_name>
/latest.pth. This makes it easy to train from
the checkpoint of a previous run since logs are
saved this way by default. Note: if you are loading a
checkpoint from a previous run, check for the name
the run in the foundation_model/train/logs/
<project_name>/, since the code appends a string
of the date to each run_name specified in the config
yaml file of the run to avoid duplicate run names.
4) Run this command inside the
foundation_model/train/ di-
rectory: python train.py -c
<path_of_train_config_file>
