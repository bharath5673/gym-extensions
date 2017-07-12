## SpaceX-like Falcon Rocket

I've assembled a Falcon rocket that can learn to land itself on the autonomous floating drone ship. I reused a part of the LunarLander (literally, the lander) from <a href="https://github.com/openai/gym">OpenAI Gym</a> and added some more elements so that it makes the task more difficult and more interesting.

The state-dimensions contains now 13 params, some of which are part of the drone-ship. The difficulty of the task can be managed in many ways (e.g., adding more/less power to rocket's engines, increasing/decreasing drone's force that pushes it around, so that it oscillates from left to right and back with different velocity, etc).

The reward function accounts for the falcon rocket landing on both legs (otherwise it's not quite a landing), for the angle and the velocity of the drone ship (higher speed, more unstable drone ship, higher the reward); the episodes are done once the rocket has landed, or if the rocket goes below the drone ship level.

You can land the SpaceX Falcon rocket from the keyboard as well.

[![Take a look at it here ](https://github.com/vBarbaros/gym-extensions/raw/master/assets/Falcon.png)](https://www.youtube.com/watch?v=sg-6g9DtfrY&feature=youtu.be "Hope you'll like it!")