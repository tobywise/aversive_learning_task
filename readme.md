# Aversive learning task

To run the task:

`python aversive_learning_task.py`

Settings are available in `aversive_learning_task_settings.yaml`

Subjects play the role of an explorer entering abandoned mines and trying to find precious rocks. Some of the rocks will have snakes hiding inside them, and the subject has to collect the rocks while avoiding snakes - finding a snake gives a shock.

There are four different mine entrances (distinguished by colour), 2 of which lead to purple rocks and 2 of which lead to pink rocks. The purple and pink rocks have a varying probabiliy of shock.

For now, the shocking component isn't complete. Extra code will be needed to administer the shocks, and if we want to give shocks at the end of a block (rather than every trial) we'll need to add that in.

## Task structure

1. Transition training - learning which of the four mine entrances lead to which of the two rock types
2. Transition test - testing subjects' knowledge of these transitions. If subjects pass the transition test (requires 80% correct, modifiable in settings), they move on to the practice.
3. Practice - subjects learn to play the real task. One type of rock is 100% associated with shock, the other is 0%.
4. Real task