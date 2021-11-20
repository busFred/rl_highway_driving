* 00.json: initial
* 01.json: only increase number of training episodes; observe instability after 800 episodes.
* 02.json: decrease discount factor from 0.9 to 0.7 and hope the agent care more about immediate reward/cost; the rest remains as same as 01.json
* 03.json: reduce epsilon decay rate from 0.99 to 0.9 and decrease the update decay rate period from 20 episodes to 10 episodes; the rest remains as same as 01.json