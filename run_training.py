###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import configparser
import time

start = time.time()
config_name = None
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    print("Wrong Augment!")
    exit(1)

# config file to read from
config = configparser.RawConfigParser()
config.readfp(open(r'./' + config_name))
# ===========================================
# name of the experiment
name_experiment = config.get('experimentname', 'name')
nohup = config.getboolean('trainingsettings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '
print(run_GPU)
# create a folder for the results
result_dir = name_experiment
print("\n 1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing")
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print("copy the configuration file in the results folder")
if sys.platform=='win32':
    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')



# run the experiment
if nohup:
    print("\n 2. Run the training on GPU with nohup")
    os.system(run_GPU +' nohup python -u ./src/retina_unet_training.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
else:
    print("\n 2. Run the training on GPU (no nohup)")
    os.system(run_GPU +' python ./src/retina_unet_training.py')
 
# Prediction/testing is run with a different script
