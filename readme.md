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

    * `drl_algs`
        
        The package that contains the drl algorithms.

    * `drl_utils`
        
        The package that contains the utilities for drl.

    * `int_mpc/`
        
        The package that defines the int_mpc Markov Decision Process.

    * `mdps/`

        The package that contains base classes for Markov Decision Process.

        * `mdp_abc`

            Defines base classes `State`, `Action`, and `Environment`.
    
        * `mdp_utils`
    
            Contains helper functions specifically for mdp.
    
        * `policy_abc`
            
            WORK IN PROGRESS.
    
            Contains classes that define the policy. Some RL algorithm requires independent policy ML models; this module is intended for this purpose.
    
            However, as this is more likely to be used with PyTorch, it might be moved to `int_mpc.nets` module. NOT DETERMINED YET.