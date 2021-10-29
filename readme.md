# Simplified IntMPC

## Install
* Develop Mode

    Install the package in development mode
        
        cd ~/path/to/csce_790_final_proj
        pip install -e .
    
## Folder Structure

* `notebook/`

    Contains all the tiny little demonstration code or test jupyter notebook file.

* `script/`
    
    Contains all the files used to train the nnet.

* `src/`
    * `int_mpc/`
        
        The base package that contains all the needed subpackage, functions, and classes required for building the simplified IntMPC.

## IntMPC Package Description:

* `mdps`
    
    This subpackage contains all the modules related to defining the Markov Decision Process.
    
    * `mdp_abc`
        
        Defines base classes `State`, `Action`, and `Environment`.

    * `mdp_utils`

        Contains helper functions specifically for mdp.

    * `policy_abc`
        
        WORK IN PROGRESS.

        Contains classes that define the policy. Some RL algorithm requires independent policy ML models; this module is intended for this purpose.

        However, as this is more likely to be used with PyTorch, it might be moved to `int_mpc.nets` module. NOT DETERMINED YET.

* `nets`
    
    This subpackage contains all the function and classes used to define drl models. Might rename this to `drl_alg` and let `nets` contains module definition.

* `utils`

    Contains more general purpose utility functions or classes.