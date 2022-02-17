# PyDFTele
The Density Functional Theory for Electrolyte Solutions

<!---
$$ \Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{id}[\{\rho_i(\boldsymbol{r})\}] + F_{exc}[\{\rho_i(\boldsymbol{r})\}]+ F_{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] + \sum_i \int_{V} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r}) d \boldsymbol{r}+ \int_{\partial V}\sigma(\boldsymbol{r}) \psi(\boldsymbol{r})  d \boldsymbol{r} $$)
-->

<a href="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5COmega%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%2C%5Cpsi%28%5Cboldsymbol%7Br%7D%29%5D%20%3D%5C%20%26%20F%5E%5Ctext%7Bid%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20&plus;%20F%5E%5Ctext%7Bexc%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D&plus;%20F%5E%5Ctext%7Bcoul%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%2C%5Cpsi%28%5Cboldsymbol%7Br%7D%29%5D%20%5C%5C%20%26%20&plus;%20%5Csum_i%20%5Cint_%7BV%7D%20d%20%5Cboldsymbol%7Br%7D%20%5BV_i%5E%7B%28%5Ctext%7Bext%7D%29%7D%28%5Cboldsymbol%7Br%7D%29-%5Cmu_i%5D%20%5Crho_i%28%5Cboldsymbol%7Br%7D%29&plus;%20%5Cint_%7B%5Cpartial%20V%7Dd%20%5Cboldsymbol%7Br%7D%20%5Csigma%28%5Cboldsymbol%7Br%7D%29%20%5Cpsi%28%5Cboldsymbol%7Br%7D%29%20%5Cend%7Balign*%7D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5COmega%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%2C%5Cpsi%28%5Cboldsymbol%7Br%7D%29%5D%20%3D%5C%20%26%20F%5E%5Ctext%7Bid%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20&plus;%20F%5E%5Ctext%7Bexc%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D&plus;%20F%5E%5Ctext%7Bcoul%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%2C%5Cpsi%28%5Cboldsymbol%7Br%7D%29%5D%20%5C%5C%20%26%20&plus;%20%5Csum_i%20%5Cint_%7BV%7D%20d%20%5Cboldsymbol%7Br%7D%20%5BV_i%5E%7B%28%5Ctext%7Bext%7D%29%7D%28%5Cboldsymbol%7Br%7D%29-%5Cmu_i%5D%20%5Crho_i%28%5Cboldsymbol%7Br%7D%29&plus;%20%5Cint_%7B%5Cpartial%20V%7Dd%20%5Cboldsymbol%7Br%7D%20%5Csigma%28%5Cboldsymbol%7Br%7D%29%20%5Cpsi%28%5Cboldsymbol%7Br%7D%29%20%5Cend%7Balign*%7D" title="\begin{align*} 
\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] =\ & F^\text{id}[\{\rho_i(\boldsymbol{r})\}] + F^\text{exc}[\{\rho_i(\boldsymbol{r})\}]+ F^\text{coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] \\
& + \sum_i \int_{V} d \boldsymbol{r}  [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r})+ \int_{\partial V}d \boldsymbol{r} \sigma(\boldsymbol{r}) \psi(\boldsymbol{r}) 
\end{align*}" /></a>

The Coulomb's free-energy <img src="https://latex.codecogs.com/svg.image?F^\text{coul}" title="F^\text{coul}" /> is obtained by the addition of the electric field energy density and the minimal-coupling of the interaction between the electrostatic potential <img src="https://latex.codecogs.com/svg.image?\psi(\boldsymbol{r})" title="\psi(\boldsymbol{r})" /> and the charge density <img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_i&space;Z_i&space;e\rho_i(\boldsymbol{r})" title="\inline \sum_i Z_i e\rho_i(\boldsymbol{r})" />, and it can be written as 

<img src="https://latex.codecogs.com/svg.image?F^\text{coul}&space;=&space;-\int_V&space;d\boldsymbol{r}\&space;\frac{\epsilon_0&space;\epsilon_r}{2}&space;|\nabla{\psi(\boldsymbol{r})}|^2&space;&plus;&space;&space;\int_{V}&space;d\boldsymbol{r}\&space;\sum_i&space;Z_i&space;e&space;\rho_{i}(\boldsymbol{r})&space;\psi(\boldsymbol{r})" title="F^\text{coul} = -\int_V d\boldsymbol{r}\ \frac{\epsilon_0 \epsilon_r}{2} |\nabla{\psi(\boldsymbol{r})}|^2 + \int_{V} d\boldsymbol{r}\ \sum_i Z_i e \rho_{i}(\boldsymbol{r}) \psi(\boldsymbol{r})" />

where <img src="https://latex.codecogs.com/svg.image?Z_i" title="Z_i" /> is the valence of the ion i, <img src="https://latex.codecogs.com/svg.image?e" title="e" />is the elementary charge, <img src="https://latex.codecogs.com/svg.image?\epsilon_0" title="\epsilon_0" /> is the vacuum permittivity, and <img src="https://latex.codecogs.com/svg.image?\epsilon_r" title="\epsilon_r" /> is the relative permittivity.

The excess free-energy is written as
<!---
$$F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{hs}[\{\rho_i(\boldsymbol{r})\}] + F_{ec}[\{\rho_i(\boldsymbol{r})\}] $$
-->

<a href="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bexc%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20%3D%20F%5E%5Ctext%7Bhs%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20&plus;%20F%5E%5Ctext%7Bec%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bexc%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20%3D%20F%5E%5Ctext%7Bhs%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D%20&plus;%20F%5E%5Ctext%7Bec%7D%5B%5C%7B%5Crho_i%28%5Cboldsymbol%7Br%7D%29%5C%7D%5D" title="F^\text{exc}[\{\rho_i(\boldsymbol{r})\}] = F^\text{hs}[\{\rho_i(\boldsymbol{r})\}] + F^\text{ec}[\{\rho_i(\boldsymbol{r})\}]" /></a>

The electrostatic correlation <a href="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F%5E%5Ctext%7Bec%7D" title="F^{\textrm{ec}}" /></a> can be described using different approximations as
- bulk fluid density (BFD) [Kierlik and Rosinberg, Phys.Rev.A 44, 5025 (1991); Y. Rosenfeld, J. Chem. Phys. 98, 8126 (1993)]
- functionalized mean spherical approximation (fMSA) [Roth and Gillespie, J. Phys.: Condens. Matter 28, 244006 (2016)]
