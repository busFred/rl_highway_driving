# Simplified IntMPC

## Install
* Develop Mode

    Install the package in development mode
        
        cd ~/path/to/csce_790_final_proj
        pip install -e .

    Note that two packages have dependency conflict, which will cause runtime error. Use `pip` to downgrade the package would resolve the problem.

## Usage

* Test Random

        python 

* Train
    
        python 

* Test Trained DQN

        python 
    
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
