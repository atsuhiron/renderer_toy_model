# renderer_toy_model
Rendering 3D CG.

## Requirement
Python 3.12

## Installation
To use it, execute the following command in a directory available `python` command,
```commandline
pip install -r requirements.txt
```
Or, when developing, execute the following commands,
```commandline
pip install -r requirements_dev.txt
```

## Usage
The following commands can be used to render.
```commandline
python main.py "samples/simple_world.json" -g 3 -c 6
```
where `-g` represents the maximum number of generations of particles to be traced, 
and `-c` is the number of children produced by one parent particle.

> [!CAUTION]
> Note the computation time. It is approximately follows the formula:  
> $t\propto P(S_r c^g + S_s)$  
> where $P$ is the number of pixel, $S_r$ is the number of rough surface, $S_s$ is the number of smooth surface, respectively.