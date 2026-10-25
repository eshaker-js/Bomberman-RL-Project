# History

## 2026-10-25 — First playable environment

Basic version of Bomberman, currently starts with a pre-determined 11x11 grid.

The grid has 3 types of cells:
Empty
Wall = unwalkable terrain.
Block = breakable terrain (with a bomb)

The game takes place in a class called BombermanEnv.
The class keeps data on the following:

1. Grid State.
2. Player Location
3. If the player is dead.
4. Bomb Location
5. Explosion (active cells marked as explosion that kill the player if he is in them)
6. Number of steps

At every turn the player can take an action:
1. Move (U,D,L,R)
2. Plant Bomb
3. Wait

The bomb is a class that keeps track of the bombs location and its timer. Currently a bomb is placed with a timer value of 4 (steps).

After every action/step, the env class returns the current state including all the data stated above^

Cursor also generated a file "play.py" that allows you to run the environment as a game and try it for yourself.
Controls: WASD/ARROWKEYS - Move | Space - Bomb | N or . - Wait | R - Reset | Esc - Quit 
(this is also detailed in the game window.)
I suggest running the game and experimenting after reading this.

### Next Tasks:

Right now when the player tries to walk on unwalkable terrain (block or wall) the game doesn't move the player (good) but consumes a move and doesn't provide feedback (probably bad, but might be ok, we need to decide together)

This is still a very early version, serves mainly as a skeleton we can start from.

I do think we can start running a simple RL Model and see if it can learn to move, explode a block and survive its own bomb as a first step. (that's what I'd want to do next if I'm the one progressing again)
