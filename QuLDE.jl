using Yao
using QuAlgorithmZoo
using LinearAlgebra
using OrdinaryDiffEq
using BitBasis
using Random
using Test

#Linear Diff Equation Unitary M
function diffeqProblem(nbit::Int)
    siz = 1<<nbit
    A = rand_unitary(siz)
    b = normalize!(rand(ComplexF64, siz))
    x = normalize!(rand(ComplexF64, siz))
    A, b, x
end

# Define Stuff
nbit = 1
M,b,x = diffeqProblem(nbit)
k = 3;

T = Int(round(log2(k+1)))
t = 0.4



C(m) = norm(x)*(t)^(m)/factorial(m)

D(n) = norm(b)*(t)^(n-1)*t/factorial(n)

C_tilda = 0
D_tilda = 0
for i in 1:k
    C_tilda = C_tilda + C(i)
    D_tilda = D_tilda + D(i)
end
C_tilda = C_tilda + C(0)
N = sqrt(C_tilda + D_tilda)
C_tilda = sqrt(C_tilda)
D_tilda = sqrt(D_tilda)
V = [C_tilda D_tilda; D_tilda -1*C_tilda]/N
V = convert(Array{ComplexF64,2},V)
V = matblock(V)


col1 = (1/C_tilda) * [sqrt(C(m)) for m in 0:k]
col2 = (1/D_tilda) * [m < k+1 ? sqrt(D(m)) : 0 for m in 1:k+1]
VS1 = rand(ComplexF64,k+1,k+1)
VS2 = rand(ComplexF64,k+1,k+1)
VS1[:,1] = col1
VS2[:,1] = col2
VS1 = -1*qr(VS1).Q
VS2 = -1*qr(VS2).Q
WS1 = VS1'
WS2 = VS2'
VS1 = matblock(VS1)
VS2 = matblock(VS2)
WS1 = matblock(WS1)
WS2 = matblock(WS2)

inreg = ArrayReg(x) ⊗ zero_state(T) ⊗ ( (C_tilda/N) * zero_state(1) )+  ArrayReg(b) ⊗ zero_state(T) ⊗ ((D_tilda/N) *ArrayReg(bit"1") )

n = 1 + T + nbit
M = M
circuitInit = chain(n, control((-1,),(2:T+1...,)=>VS1),control((1,),(2:T+1...,)=>VS2))
  
circuitIntermediate = chain(n)
a = Array{Int64,1}(undef, T)
U = Matrix{ComplexF64}(I, 1<<nbit,1<<nbit)
for i in 0:k
    digits!(a,i,base = 2)
    G = matblock(U) 
    push!(circuitIntermediate,control(n, (-1*collect(2:T+1).*((-1*ones(Int, T)).^a)...,), (T+2:n...,)=>G))
    U = M*U
end

circuitFinal = chain(n, control((-1,),(2:T+1...,)=>WS1),control((1,),(2:T+1...,)=>WS2), put(1=>V))

res = apply!(inreg,chain(circuitInit,circuitIntermediate,circuitFinal)) |> focus!(1:T+1...,) |> select!(0) |> state

out = (N^2)*(vec(res))

f(u,p,t) = M*u + b;
tspan = (0.0,0.4)
prob = ODEProblem(f, x, tspan)
sol = solve(prob, Tsit5(), dt = 0.1, adaptive = :false)
s = vcat(sol.u...)

@test isapprox.(s[end-1:end], out, atol = 0.05) |> all
