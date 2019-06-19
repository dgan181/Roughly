
using QuDiffEq
using OrdinaryDiffEq
using ForwardDiff
using Plots
"""
Linearizing non linear ODE and solving it using QuLDE

du/dt = f
u0 -> inital condition for the ode
Δu -> difference for fixed point
h -> time step
k -> order of Taylor Expansion in QuLDE circuit

Equation input to QuLDE circuit : d(Δu)/dt = fJ * Δu + b
"""

function fo(du,u,p,t)
    du[1] = -2*u[2]^2*u[1]
    du[2] = 3*u[1]^(1.5) - 0.1*u[2]
end

f(x,y) = [-2*y^2*x, 3*x^(1.5) - 0.1*y];
u0 = [0.2,0.1]
Δu = [1e-6,1e-6] 
h = 0.01
k = 3
tspan = (0.0,0.8 - h)
length = round(Int,(tspan[2] - tspan[1])/h) + 1
res = Array{Float64,2}(undef,length,2)
a = u0
for i in 1:length
    res[i,:] = a
    fJ = ForwardDiff.jacobian(u -> f(u...), a)
    b = f(a...)
    qprob = QuLDEProblem(fJ,b,Δu,(0.0,h))
    out = solve(qprob,QuLDE(),k,2)
    a = convert(Array{Real,1},out + a)   
end
prob = ODEProblem(fo,u0,tspan)
sol = solve(prob,Tsit5(),dt = h,adaptive = :false)
v = transpose(hcat(sol.u...))
t = 0.0:h:0.8-h

gr();
p1 = plot(t, [v,res], label= ["ODE_x", "ODE_y", "QuODE_x", "QuODE_y"],legend = :bottomright)


