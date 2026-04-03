# Vehicle-Scooter Interaction
## Instructions for Reproducing Results
Begin by ensuring you have Python downloaded. The dependencies required (matplotlib) can be installed by running the command `pip install matplotlib` or `pip3 install matplotlib`.

Once you have installed the dependencies and are in the "vehicle_scooter_interaction" folder, run the command `python vsi.py` or `python3 vsi.py` to run the tests for vehicle-scooter interaction.

## Interpreting the Results
The script runs two tests and has three figures of results. 

First, I test how pruning dangerous states with BPA compares to simply using a hazard penalty in the case where a scooter takes a deterministic left turn. 
In the animation "mcts_pruning_comparison.gif", I present four animated simulations comparing four MCTS models: BPA pruning (no hazard penalty), Hazard Penalty = 10.0 (no pruning), Hazard Penalty = 50.0 (no pruning), and Hazard Penalty = 100.0 (no pruning).
The red square represents the car that is making decision and the green dot represents the scooter the car needs to yield to. The yellow squares in front of the scooter represent potential future states of the scooter in its field of view. When the car reaches the goal, a colored square shows the reaction quality and the step counts to reach the goal is displayed. A red square represents colliding with the scooter; a yellow square represents changing lanes, stopping abruptly, or cutting close in front of the scooter; a green square represents none of these poor reactions and allows for a stop shortly before the intersection.  

Additionally, I test the robustness of the BPA pruning model and compare it to the robustness of a hazard penalty model by evaluating the models on a stochastic scooter. A quarter of the time, the scooter chooses it's next step from a distribution of potential next states.
In the animation, "mcts_intersection_robustness.gif", I show the first four responses of my BPA pruning model to the stochastic scooter. 
I compare the overall results of my BPA pruning model compared to the Hazard Penalty = 50.0 model in the histogram "robustness_steps_histogram_comparison.png".
It can be seen that the BPA pruning model is significantly better than the hazard penalty model at avoiding hazardous outcomes and handling them well. This comes at a tradeoff of more time to reach the goal, but is preferred over dangerous driving policies.
I believe the pruning method is advantageous over the penalty method because we can tune a parameter for a time/distance buffer from the scooter state and generally have a guarantee that no collision will occur. Collisions may occur if the scooter crosses the median and all future states are pruned for the car. Hazard penalties, on the other hand, require more relative tuning to other parameters and does not have the explainable guarantees that BPA pruning offers.

## Disclaimer
I used Github Copilot as a productivity aid in developing my code.  
Inspired by the work of Zhitong He et al., "Risk Analysis in Vehicle and Electric Scooter Interaction".