# msviz

A package created to visualize my masters results in the form of graphs, using julia with dataframes.jl and plots.jl.  
This also acts as a place to store my raw log files.

This package also stores some important checkpoints they can be run using [ScalableHrlEs](https://github.com/sash-a/ScalableHrlEs.jl)


## How to run checkpoints

Clone [ScalableHrlEs](https://github.com/sash-a/ScalableHrlEs.jl), [ScalableES](https://github.com/sash-a/ScalableES.jl) and [HrlMuJoCoEnvs](https://github.com/sash-a/HrlMuJoCoEnvs.jl) and install the packages using the julia package manager (`pkg> instantiate`).

Assumes msviz and ScalableHrlEs are in the same folder.

```
cd ScalableHrlEs.jl
julia --project scripts/runsaved.jl saved/remote/keep/AntFall-pt 1000 AntFall --pretrained
```

Run `julia --project scripts/runsaved.jl --help` for info on the arguments  
See the generation number of the file inside checkpoint folder to know which number to put for generation arg. ie AntFall is generation 1000 and AntFall-pt is generation 3000.  
I only saved one base SHES policy and SHES-TL policy (these are labeled with a 'pt' at the end of the folder) as checkpoints take up a lot of space. They are saved using BSON.