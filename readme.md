# Simplified IntMPC

## Install
* Develop Mode

    Install the package in development mode
        
        cd ~/path/to/csce_790_final_proj
        pip install -e .

    Note that two packages have dependency conflict, which will cause runtime error. Use `pip` to downgrade the package would resolve the problem.

## Usage

All action assumed python virtual environment is properly installed, and the terminal is changed to the location of the script.

* Quick Test with DQN Policy

    With IPython Kernel, the environment and the load a trained DQN policy can be run cell by cell. The script can be found at `notebook/run_dqn.py`.

* Test Random

    Generate statistics using random policy. The script is located in `script` directiory.

        python run_random.py --env_config_path ../config/change_lane/01.json --export_metrics_dir ../metrics/random --n_test_episodes 20 --to_vis

* Train
    
    Train DQN metwork. The script is located in `script` directiory.
    
        python train_dqn.py --env_config_path ../config/change_lane/01.json --dqn_config_path PATH/TO/DQN_CONFIG --export_path ../model/ --lr 1e-5 --n_val_episodes 20 --max_workers 7 --checkpoint_path ../checkpoint --metrics_path ../metrics/train/


* Test Trained DQN
    
    Generate statistics using DQN policy. The script is located in `script` directiory.

        python run_dqn.py --env_config_path ../config/change_lane/01.json --dqn_path ../model/env_01_dqn_02.pt --export_metrics_dir ../metrics/dqn/env_01_dqn_02 --n_test_episodes 20 --use_cuda --max_workers 7 --to_vis
    
## Folder Structure

* `notebook/`

    Contains all the tiny little demonstration code or test jupyter notebook file.

* `script/`
    
    Contains all the files used to train the nnet.

* `src/`

    * `drl_algs`
        
        The package that contains the drl algorithms.

        * `dqn`

            The deep q learning algorithm.

    * `drl_utils/`
        
        * `buff_utils`
            
            Defines replay buffer.

    * `int_mpc/`
        
        The package that defines the int_mpc Markov Decision Process.

        * `mdps`
            
            Defines the State for highway env 

        * `change_lane`

            Defines the change lane environment.

        * `nnet`
        
            `dqn`: The definition of the network itself.

    * `mdps/`

        The package that contains base classes for Markov Decision Process.

        * `mdp_abc`

            Defines base classes `State`, `Action`, and `Environment`.
    
        * `mdp_utils`
    
            Contains helper functions specifically for mdp, such as used the trained policy to run the simulation.
