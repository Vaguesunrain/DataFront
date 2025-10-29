# lms vs nlms vs vss-nlms

## lms
![alt text](image.png)

    --- Running on ideal floating-point signals ---
    Normalized signal power: 0.1600
    Calculated step size (mu): 0.0391

    --- Performance Evaluation ---
    SNR Before: 0.51 dB
    SNR After:  8.68 dB
    SNR Improvement: 8.17 dB
    Theoretical Max Improvement: 17.01 dB

## nlms
![alt text](image-1.png)

    --- Running NLMS on ideal floating-point signals ---
    Using NLMS algorithm with mu = 0.20

    --- Performance Evaluation ---
    SNR Before: 0.51 dB
    SNR After:  8.25 dB
    SNR Improvement: 7.73 dB
    Theoretical Max Improvement: 17.01 dB

## vss-nlms
![alt text](image-2.png)

    
    SNR Before:       0.51 dB
    SNR After:        15.62 dB
    SNR Improvement:  15.10 dB
    Theoretical Max:  17.01 dB
    Efficiency:       88.8%
    Steady-state MSE: 1.85e-01
    Excess MSE:       1.81e-01
    ===================

    正在运行标准 NLMS 进行对比...
    SNR (Fixed NLMS): 10.94 dB
    VSS-NLMS Gain:    4.68 dB


### provement

input:  $x[n] , d[n]$ : $x$ is the noise, $d$  is the noise + signal we need(desirable).
weight:  $w[n]$.
output:  $y = w[n]^T * x $,$e$ = $d$ - $y$ 

we hope the 'e' can be as small as possible.in other word, Our goal is to minimize the error power, $J = E[e^2]$.

we need calculate the gradient of J with respect to w：
$∇J = ∂J/∂W = E[2 * (d(n) - Wᵀx(n)) * (-x(n))] = -2 * E[e(n) * x(n)]$

but we can't calculate true $E[e(n)]$ , so we use an approximation(use the instantaneous value at the current time): 
$∇J ≈ -2 * e(n) * x(n)$
$
W(n+1) = W(n) - (small step) * ∇J
$
$W(n+1) = W(n) + (small step) * 2 * e(n) * x(n)$
let  $(small step) * 2 = (Step Size) μ $
$
W(n+1) = W(n) + μ * e(n) * x(n)
$

above all is the provement of LMS.

### The defect: $w(n)$ calculate will be influenced by $e(n)$ and $x(n)$.
 from $μ * e(n) * x(n)$,we can see that the weight calculate will be influenced by $e(n)$ and $x(n)$. we try deminishing the influence caused by $x(n)$ (Normalization).
 $||x(n)||² = x(n)ᵀx(n)$
 $W(n+1) = W(n) + (μ / (||x(n)||² + ε)) * e(n) * x(n)$
$epsilon$ is a very small number,to prevent the signal power is 0.

this method makes the changing more controllable.

### the step is a constant ,we don't know if it is suitable.
if $e(n)$ is large,we should let $μ$ more lager(we need convergence),otherwise smaller(we need accuracy).
someone formulate the formulation: $μ(n+1) = α * μ(n) + γ * e(n)²$
