#!/bin/bash

# start.sh -- Convenient bash script to start the training/testing
#             Just tweak the parameters in the given variables accordingly
#
# NOTE: Make sure the data/ directory is populated accordingly
#       Copy-paste the .wav files into their corresponding directories

################################ WARNING: Don't change these values ################################
PYTHON=python3              # Change the value to "python" in case python3 doesn't work
MAIN_SCRIPT=main.py         # The entry point for the program

PRETRAIN_DIR=pretrained     # The directory to store the pretrained models
LOG_DIR=logs                # The directory to store the log files for tensorboard
CONFIG_JSON=config.json     # The configuration script holding the information about the parameters
####################################################################################################

#------------------------------------------
# The model to use for training
# Select any one of the following
#   - ann
#   - lstm
#   - gru
#   - dqn         (Not supported yet)
#   - reinforce   (Not supported yet)
#
# As for the parameters for the model, see
# config.json file. Change the parameters
# there to instead of tweaking the code
#------------------------------------------
MODEL_TO_USE=ann

#------------------------------------------
# Start the program in which mode ?
#   - train
#   - test         (Not supported yet)
#------------------------------------------
RUNNING_MODE=train


# Step-1: Remove the previously saved pretrained models, if any
#         Create the directory if not already created
if [ ! -d "$PRETRAIN_DIR" ]; then
  mkdir $PRETRAIN_DIR
else
  rm -rf ./"$PRETRAIN_DIR"/*.pkl
fi

# Step-2: Remove the previously saved logs, if any
#         Create the directory if not already created
if [ ! -d "$LOG_DIR" ]; then
  mkdir $LOG_DIR
else
  rm -rf ./"$LOG_DIR"/*
fi

# Everything is ready. Start !
$PYTHON $MAIN_SCRIPT --config $CONFIG_JSON \
                     --model $MODEL_TO_USE \
                     --mode $RUNNING_MODE