# PyDFTele
The Density Functional Theory for Electrolyte Solutions

<!---
$$ \Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{id}[\{\rho_i(\boldsymbol{r})\}] + F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]+ \sum_i \int_{V} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r}) d \boldsymbol{r}- \int_{\partial V}\sigma(\boldsymbol{r}) \psi(\boldsymbol{r})  d \boldsymbol{r} $$)
-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=&space;F_{id}[\{\rho_i(\boldsymbol{r})\}]&space;&plus;&space;F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&plus;&space;\sum_i&space;\int_{V}&space;[V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i]&space;\rho_i(\boldsymbol{r})&space;d&space;\boldsymbol{r}-&space;\int_{\partial&space;V}\sigma(\boldsymbol{r})&space;\psi(\boldsymbol{r})&space;d&space;\boldsymbol{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=&space;F_{id}[\{\rho_i(\boldsymbol{r})\}]&space;&plus;&space;F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&plus;&space;\sum_i&space;\int_{V}&space;[V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i]&space;\rho_i(\boldsymbol{r})&space;d&space;\boldsymbol{r}-&space;\int_{\partial&space;V}\sigma(\boldsymbol{r})&space;\psi(\boldsymbol{r})&space;d&space;\boldsymbol{r}" title="\Omega[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{id}[\{\rho_i(\boldsymbol{r})\}] + F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]+ \sum_i \int_{V} [V_i^{(\text{ext})}(\boldsymbol{r})-\mu_i] \rho_i(\boldsymbol{r}) d \boldsymbol{r}- \int_{\partial V}\sigma(\boldsymbol{r}) \psi(\boldsymbol{r}) d \boldsymbol{r}" /></a>

The excess free-energy is written as
<!---
$$F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{hs}[\{\rho_i(\boldsymbol{r})\}]+ F_{Coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] + F_{ele-corr}[\{\rho_i(\boldsymbol{r})\}] $$
-->

<a href="https://www.codecogs.com/eqnedit.php?latex=F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=&space;F_{hs}[\{\rho_i(\boldsymbol{r})\}]&plus;&space;F_{Coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;&plus;&space;F_{ele-corr}[\{\rho_i(\boldsymbol{r})\}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;=&space;F_{hs}[\{\rho_i(\boldsymbol{r})\}]&plus;&space;F_{Coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})]&space;&plus;&space;F_{ele-corr}[\{\rho_i(\boldsymbol{r})\}]" title="F_{exc}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] = F_{hs}[\{\rho_i(\boldsymbol{r})\}]+ F_{Coul}[\{\rho_i(\boldsymbol{r})\},\psi(\boldsymbol{r})] + F_{ele-corr}[\{\rho_i(\boldsymbol{r})\}]" /></a>

The electrostatic correlation <a href="https://www.codecogs.com/eqnedit.php?latex=F_{\textrm{ele-corr}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{\textrm{ele-corr}}" title="F_{\textrm{ele-corr}}" /></a> can be described using different approximations as
- bulk fluid density (BFD) [Kierlik and Rosinberg, Phys.Rev.A 44, 5025 (1991); Y. Rosenfeld, J. Chem. Phys. 98, 8126 (1993)]
- functionalized mean spherical approximation (fMSA) [Roth and Gillespie, J. Phys.: Condens. Matter 28, 244006 (2016)]
