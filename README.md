# Models used in: {Title manuscript}
This repo contains the model and the parameters used for the manuscipt {title manuscript} in the journal phiRO (DOI:).
Final model parameters have been set as default. 

## Summary: 
To alleviate computational burden of intensity-modulated proton therapy beam-angle optimisation (BAO) [1], we automatically generated BAO plans for 310 oro- and hypopharyngeal patients treated at our institute to (i) propose a new BAO derived class solution (ii) train a deep-learning  model to predict favourable beam angles.

BAO [1] was used to select for each patient four beam directions from 72 equidistant angles. Median angles were used to define the class solution. For DL, patients were divided into training (248), validation (31) and test(31) sets, stratified on the median beam directions. Two types of input were used: 
    1. Volumetric: Normalised CT data,  structure set, isocentric distance maps [2] 
    2. Scalar: OAR overlap volume fractions (as used in BAO cost-functions [1]) 
U-Nets with different depths and widths (constrained by NVIDIA A100 memory) and volumetric input were fused with dense layers that used the scalar input. We investigated: (i) prediction of normalised angles using cosine-based loss, and (ii) prediction of normalised Cartesian coordinates on the unit circle using squared Euclidian-distance loss. Model selection was performed based on the validation set, and trained using 10 different initialisations (seeds) with batch size 5. The best three initialisations based on validation loss were ensembled by averaging the outputs. Data was augmented by left-right flipping. Wish-list based automated treatment planning in Erasmus-iCycle [3,4] was used to generate final robust IMPT plans for all 31 test patients for: 1) the BAO derived class solution 2) deep learning predicted beam angles and 3) the clinically used 4-field class solution (angles: 50, 150, 210, 310). After scaling to the same target coverage, NTCPs were calculated [5].

References 
[1] Kong et al. 2025, DOI: 10.1016/j.radonc.2025.110799
[2] Nomer et al. 2024, DOI: 10.1088/1361-6560/ad8c95
[3] Breedveld 2012, DOI: 10.1118/1.3676689
[4] Kong 2024, DOI: 10.1088/1361-6560/ad1e7a
[5] Langendijk et al. 2021, DOI: 10.14338/IJPT-20-00089.1
