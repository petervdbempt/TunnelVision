# TunnelVision Environment
TunnelVision is a hard-exploration reinforcement learning task based on the Gymnasium project. Below is a representation of the standard world.
![TunnelVision](https://github.com/petervdbempt/TunnelVision/assets/104323782/eda4a9ea-226a-45ce-856a-b6e23767312d=480x270)
The starting position of the agent is in the top-left corner, and its goal is to reach the treasure located in the bottom-right corner. The environment is filled with toxic gasses that immediately terminate an episode when encountered. Additionally, there are two inferior rewards which also end the episode.

# Local Reward Maxima
The TunnelVision environment contains multiple local reward maxima, represented as inferior rewards. The environment is meant to demonstrate the importance of separating extrinsic and intrinsic rewards. You will see that many standard algorithms get attracted to inferior rewards and never find the treasure.

# Additional Information
The action space for the agent is discrete (up, down, right, left) and represented by integers 0-3. The state space for the standard world also discrete; it is a grid space with size 4x12. The reward is zero for reaching toxic gas states, 0.25 for reaching inferior reward states and 1 for reaching the treasure.

# Baselines
Included in the project are five algorithms to make comparisons with: 
 Markup : * Q-Learning
              * Regular Q-Learning with exponential epsilon-decay
          * Q-Switching_Blind
              * Q-Learning algorithm that switches modes using a fixed step interval
          * Q-Switching_Blind_Separate
              * Q-Learning algorithm that switches modes using a fixed interval, and separates extrinsic rewards from intrinsic rewards
          * Q-Switching_Informed
              * Q-Learning algorithm that switches modes using a heuristic
          * Q-Switching_Blind_Separate
              * Q-Learning algorithm that switches modes using a heuristic, and separates extrinsic rewards from intrinsic rewards
