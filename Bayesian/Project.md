
### 1. Derive analytically the posterior of the vaccination coverage per poverty level. Use a conjugate prior that (1) reflects no knowledge on the vaccination coverage, and (2) reflects that vaccination coverage is typically around 90% or higher. Give posterior summary measures of the vaccination coverage per age and region. Is the choice of the prior impacting your results?


Given data is binomial. The likelihood can be set up as follows:
$$
Y_{ij} \mid \pi_{ij} \sim Bin(N_{i,j}, \pi_{i,j})
$$
where, 
- $Y_{ij}$​ = number of vaccinated children
- $N_{ij}$ = sample size (total children surveyed)
- $\pi_{ij}$  = vaccination coverage (probability of vaccination)

For the data $Y_{ij}$ distributed as Binomial, the  conjugate prior for  $\pi_{ij}$ parameter, i.e.  $\pi_{ij}$ is given by Beta distribution, 
$$
 \pi_{ij}\sim Beta(\alpha, \beta)
$$
The posterior $\pi_{ij} \mid Y_{ij}$ is given as,
$$
\pi_{ij} \mid Y_{ij} \sim Beta(\alpha+\sum y_{ij}, \beta+\sum N_{ij}-\sum y_{ij})
$$


1. For Non-informative prior, the Beta(1,1)  is selected, as it imparts no information. The Posterior then becomes $Beta(1+\sum y_{ij}, 1+\sum N_{ij}-\sum y_{ij})$
2. 
	
	