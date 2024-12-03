# Power Consumption-Aware 5G Edge UPF Selection using Deep Reinforcement Learning

## FILES

The repo contains a lot of files, most of them are from old versions or contain small variations which are not used in the final version of the work. The important files are:
*	nfv_allocation_cpu_env_rew.py 
*	env_utils.py
*	5qi_table.json
*	/models
*	/saved models
*	allocation_train.ipynb
*	compare_allocation.ipynb
*	allocation_test_multimodel.ipynb

## HOW TO RUN

1.	Modify the environment defined in `nfv_allocation_cpu_env_rew.py` and `env_utils.py` to fit the desired scenario to be emulated.
2.	Modify the training parameters defined in `allocation_train.ipynb` and run all cells to train the DRL agent. The trained agent is saved in the saved models directory.
3.	Use `compare_allocation.ipynb` to evaluate the performance of the trained DRL model against the other heuristics
4.	Use `allocation_test_multimodel.ipynb` to compare the performance of two or more trained DRL models. This is used to compare the model trained on power consumption data and CPU load.
