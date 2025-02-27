# Power Consumption-Aware 5G Edge UPF Selection using Deep Reinforcement Learning
![system_presentation](https://github.com/user-attachments/assets/99cdbb1a-3abb-4792-916f-ec85b63061b4)

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

## PUBLICATION
Please cite our published paper if you intend on using the data or code in this repository

> A. Bellin, N. Di Cicco, D. Munaretto and F. Granelli, "Power Consumption-Aware 5G Edge UPF Selection using Deep Reinforcement Learning," 2024 IEEE Conference on Network Function Virtualization and Software Defined Networks (NFV-SDN), Natal, Brazil, 2024, pp. 1-6, doi: 10.1109/NFV-SDN61811.2024.10807472.
