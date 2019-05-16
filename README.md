# chainofstate

With this tool, you can use chain-of-state methods for describing minimum energy path (MEP) with different Quantum Mechanism engines.

This script support:

- Algorithm
  - String method
  - Simple string method
  - Nudged elastic band (NEB)
  - Climbing-image nudged elastic band (CI-NEB)

- QM engine
  - Gaussian


```
{
    "coord":"init.xyz", # Filename of your input coordinates. At least contain 2 conformations.
    
    "index":[0, 2, 4, 6, 8, 10, 12, 14, 16], # The indexes of your input conformations. The middle 
                                             # coordinates between two given images will be automaticly
                                             # setted. If there are 2 or 3 given conformations, linear
                                             # interpolation will be used. If there are more given 
                                             # conformations, cubic interpolation will be chosen.
    
    "template":"template.gjf", # The template file for generating Gaussian input. The calculation method
                               # is setted here.
    
    "workflow":[ 
        {
            "jobname":"NEB-step1", # The name of your job. Pathway conformations will be saved in 
                                   # file named "$jobname-pathway$nstep.xyz".
                                   
            "method":"CINEB", # Algorithm is setted here. Now we support "NEB" and "CINEB". String
                              # method will be added in the near future.
                              
            "Kspring":5.0, # Force constant of NEB spring in kcal/mol/angstrom^2 unit.
          
           "LRate":0.001, # The learning rate of steepest descent. No unit. 
          
           "Rmax":0.10, # Max movement in each step. In unit of angstrom. If the largest movement
                         # is higher than Rmax, then all the movements will multiply a scale factor.
            
            "maxcycle":5 # The cycles this work will run.
        },
        {
            "jobname":"NEB-step2", # Another works could also be setted in workflow. The final conformations
                                   # of the previous work will be directly used. So you can change 
                                   # algorithm and hyper-parameters here.
            "method":"CINEB",
            "Kspring":5.0,
            "LRate":0.0005,
            "Rmax":0.05,
            "maxcycle":20
        }
    ]
}
```
