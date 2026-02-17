What do we want to say?
second order tools - 

1. Global goals: 
2. 
3.
4.

Inverse problems 

PDE-constrained optimisation problems 

Many problems in science and engineering involve determining unknown parameters that affect system behaviour through physical laws. 
Many of these are posed as PDE constrained optimisation problems such as topology optimisation and inverse problems. 

In this work, we introduce a framework for performing pde constrained optimisation considering arbitrary state equations and objective formulations. The design 

For many problems, the second order information can also be exploited to accelerate convergence. We also allow for the computation of second order derivatives by 
We also introduce the capabilities for performant second order optimisation methods. 

why Gridap.jl:





PDE constrained optimisation with Gridap.jl

Many problems in science and engineering involve determining unknown parameters that affect system behaviour through physical laws, and are naturally formulated as PDE-constrained optimisation problems. Important examples of such problems include topology optimisation and inverse problems. 

Gradient-based optimisation methods are the go to methods for these problems because of the large design spaces and costly objective function evaluations which involve solving sets of pdes. 

Even with gradient based methods, when dealing witrh multiphysics systems, these problems are difficult to solve becuase eof both the large costs associated with many simulation runs and the complexities around computing gradients, for example traditionally the derivatives are found analytically, often with difficulty. 

To be able to solve general multiphysics problems, we consider that the framework should adhere to the following design philosophy: Easily express multiphysics problem without the need for much problem specific work and also be fast and scalable. 

The Julia package Gridap.jl is rarara. 
This gives an expressive API that is 1:1 with the maths. We implement performant autodiff rules so that users need not specificy gradients. -- ususally the cost of a gradient eval is roughly the cost of a simulation.

FOr the efficiancy part: gridap is efficient already. We pair this with fast and scalable memnory distributed gradient (we keep in with the main design philosphies of gridap). 

Also, there are many problems that benefit from the use of second order information. We include this by including ways to automatically compute hessian vector products that could be used in some sort of newton-krylov method. 

We present some examples, e.g. 

FSI w CutFEM ... notagbly the parameters here are deeply embedde in the code - they have a complicated relationship with the integral evaluation - still do efficiantly
Contact shape reconstruction with piezoresistive grippers - Efficiantly exploit second order information in an inverse problem 





PDE-constrained optimisation for multiphysics problems with Gridap.jl

Many problems in science and engineering involve determining unknown parameters that affect system behaviour through physical laws, and are naturally formulated as PDE-constrained optimisation problems. Important examples include topology optimisation, where material distributions are designed to optimise performance, and inverse problems, where parameters are inferred from observations.

Gradient-based optimisation methods are the standard approach for such problems due to the high dimensionality of the design space and the high cost of objective function evaluations, which require the solution of one or more PDE systems. Even with gradient-based methods, PDE-constrained optimisation remains challenging for multiphysics problems, owing to the computational cost of repeated large scale simulations and the complexities of developing problem specific code, especially for computation of the gradient. In many existing approaches, gradients are derived analytically, which is often challenging for multiphysics problems, or rely on finite-differencing when derivatives become complicated [1]. 

To address these challenges, we develop a PDE-constrained optimisation framework designed to support multiphysics problems with minimal problem-specific effort and with efficiency and scalability as core design objectives. The framework is built on the Julia finite-element library Gridap.jl [2], which combines performance comparable to statically compiled code with a high-level, expressive API.

In alignment with the design principles of Gridap.jl, we provide a syntax to specify arbitrary state equations and objective functions that is near one-to-one with the mathematical notation and implement automatic differentiation tools to eliminate the requirement for user defined gradients. Using an efficient mixed-mode automatic differentiation routine, the cost of a gradient evaluation using the software is comparable to that of a function evaluation for typical cases. For problems where second-order information is beneficial, the framework also supports efficient computation of Hessian–vector products using automatic differentiation, without additional user effort.

Finally, we demonstrate the approach on representative problems, including the topology optimisation a fluid–structure interaction problem using the CutFEM and a contact shape reconstruction problem for a piezoresistive gripper. These applications highlight the ability of the framework to address complex multiphysics PDE-constrained optimisation problems from different domains. 

[1] Lorenz T. Biegler et al. “Large-Scale PDE-Constrained Optimization”. In: Lecture
Notes in Computational Science and Engineering (2003). doi: 10.1007/978-3-642-
55508-4. url: http://dx.doi.org/10.1007/978-3-642-55508-4.

[2] Santiago Badia and Francesc Verdugo. “Gridap: An extensible Finite Element toolbox
in Julia”. In: Journal of Open Source Software 5.52 (2020), p. 2520. doi: 10.21105/
joss.02520. url: https://doi.org/10.21105/joss.02520.


Reduced to 1500 characters....


Many problems in science and engineering involve determining unknown parameters that affect system behaviour through physical laws, and are naturally formulated as PDE-constrained optimisation problems. Even with gradient-based methods, PDE-constrained optimisation remains challenging for multiphysics problems, owing to the computational cost of repeated large scale simulations and the complexities of developing problem specific code, especially for computation of the gradient. In many existing approaches, gradients are derived analytically, which is often challenging for multiphysics problems, or rely on finite-differencing when derivatives become complicated. 
To address these challenges, we develop a PDE-constrained optimisation framework designed to support multiphysics problems with minimal problem-specific effort and with efficiency and scalability as core design objectives. The framework is built on the Julia finite-element library Gridap.jl [Badia and Verdugo, Gridap: An extensible Finite Element toolbox in Julia, 2020].
In alignment with the design principles of Gridap.jl, we provide a syntax to specify arbitrary state equations and objective functions that is near one-to-one with the mathematical notation and implement automatic differentiation tools to eliminate the requirement for user defined gradients. We demonstrate the approach on representative multiphysics problems.



Finally, we demonstrate the approach on representative problems, including the topology optimisation a fluid–structure interaction problem using the CutFEM and a contact shape reconstruction problem for a piezoresistive gripper. These applications highlight the ability of the framework to address complex multiphysics PDE-constrained optimisation problems from different domains. 
