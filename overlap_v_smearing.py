import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def fitfunc(p, x):
    return np.multiply(p[0],x) + p[1]
def residual(p, x, y, err):
    return (fitfunc(p, x)-y)/err

smear = [1.0, 0.5, 0.2, 0.1, 0.0]
overlap = [0.3636, 0.2929, 0.2727, 0.2828, 0.2525] 

do = np.sqrt(np.multiply(overlap,100))
print do
error = do/100

p01 = [0.14, 0.25]

pf1, cov1, info1, mesg1, success1 = optimize.leastsq(residual, p01, args = (smear, overlap, error), full_output=1)

if success1 <= 4:
    print "Fit Converged"
    chisq1 = sum(info1["fvec"]*info1["fvec"])
    dof1 = len(smear)-len(pf1)
    pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]
    print "Converged with chi-squared ", chisq1
    print "Number of degrees of freedom, dof =",dof1
    print "Reduced chi-squared ", chisq1/dof1
    print "Inital guess values:"
    print "  p0 =", p01
    print "Best fit values:"
    print "  pf =", pf1
    print "Uncertainties in the best fit values:"
    print "  pferr =", pferr1
    print
    
ax = plt.axes()
ax.errorbar(smear, overlap, yerr=error, fmt="o")

x = np.linspace(min(smear)-0.1, max(smear)+0.1, 500)
y = fitfunc(pf1, x)

ax.plot(x, y, "r-")     

ax.set_title('Overlap of Signal and Background with Different PMT Smearings')
ax.set_xlabel('Smearing (ns)')
ax.set_ylabel('Overlap (Counts)')  

plt.show() 
