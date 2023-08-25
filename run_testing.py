###################################################
#
#   Script to execute the prediction
#
##################################################

import os, sys, time
import configparser

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
# name of the experiment!!
name_experiment = config.get('experimentname', 'name')
nohup = config.getboolean('testingsettings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results if not existing already
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)


# finally run the prediction
if nohup:
    print("\n2. Run the prediction on GPU  with nohup")
    os.system(run_GPU + ' nohup python -u ./src/retina_unet_predict.py ' + config_name +
              ' > ' + './'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
    print("\n2. Run the prediction on GPU (no nohup)")
    os.system(run_GPU + ' python ./src/retina_unet_predict.py ' + config_name)

end = time.time()
print("Running time (in sed): ", end-start)
