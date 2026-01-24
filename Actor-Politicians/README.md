### Dataset Description 
A purposive sampling approach was used to gather as many actors-turned-politicians as possible, as this group is relatively limited. Purposive random sampling was applied to select a comparable number of actors who had not entered politics. To ensure similar levels of public exposure, we imposed the following selection criteria: 

- The actor must have at least 10 acting credits. 

- The actor must have been active in the entertainment industry at any point between 2005 and 2025. 

The following information was collected for each actor: 

- years_active: Numerical; Number of years active as an actor until 2025 

- imdb_credits: Numerical; Number of TV and film acting credits 

- wins: Numerical; Total number of wins for Best Actor and Best Supporting Actor categories from prominent award-giving bodies (Filipino Academy of Movie Arts and Sciences (FAMAS), Gawad Urian, and Metro Manila Film Festival (MMFF)) 

- family: Numerical; Number of immediate family members in politics 

- education: Categorical; Levels: 0 – Not finished basic education, 1 – Basic Education, 2 – Bachelor’s Degree, 3 – Postgraduate 

- politics: Categorical, target variable; 1 – Elected politician at any point until May 2022 elections, 0 – Not elected. 

Data were obtained from and manually cross-checked across the following online sources: IMDb (for acting credits and awards), Wikipedia (for career duration, political involvement, family, and educational background), Geni.com (for identifying immediate family members) and News websites such as Inquirer, Philippine Star, PEP, etc. (for validation of other information). The original dataset had 124 observations in total. These data were then sampled into 30 observations while retaining the initial proportions- shown in the 'sampling.R' file.
