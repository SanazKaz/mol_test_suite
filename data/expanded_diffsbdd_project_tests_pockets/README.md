## Expanded DiffSBDD

We expanded the dictionary for carbons to add in different carbon hybridisations instead of elemental carbon.

Carbons were expanded from C to C_sp3, C_sp2, C_sp, C_aromatic.

Unfortunately this experiment did not yield improvement in bond length, angles, posebusters validity most likely due to data imbalance and data scarcity for such a granular task.

The folder diffsbdd: contains 100 mols generated for 100 CD2020 test pocket
The folder diffsbdd_hybridised contains generated mols for the expanded_diffsbdd 

csv_analysis results: contains posebusters outputs and summary which might need to be regenerated since failure counts dont line up with the total mols.

**remove any pdbqt files which may have accidentially been saved during small docking experiment you did with 4tos pdb for each method**

PB results using mol_fast config - find them in the /raw directory.


docking:
- i docked using autodock vina the bash script vina.sh located in bash/ folder. I used the minimised version of the mols and generated the pose. I could also use the unprocessed version, sanitise then use the local minimisation instead of score only.

