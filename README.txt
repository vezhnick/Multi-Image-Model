%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Multi Image Model for Weakly Supervised Semantic Segmentation   %
%						                    %	
%       Toolbox by Alexander Vezhnevets (ETH Zurich)                %
%     						                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

License
This software is made publicly for research use only. It may be modified and redistributed under the terms of the GNU General Public License. 

Citation
Please cite the following if you plan to use the code in your own work: 
* A. Vezhnevets, V. Ferrari, J.M. Buhmann "Weakly Supervised Semantic Segmentation with a Multi-image Model", ICCV 2011 


Tested on Matlab R2010a.

This is a "beta" version of a code for our ICCV paper. It has limited functionality and implements only the top
layer of our framework. Namely, construction of the graph structure for the pairwise potentials, inference and learning,
prediction for the test set (including global multiple kernel image metric). It does not implement:
 - oversegmentation of images into superpixels
 - feature extraction for superpixels
 - feature extraction for images
 - objectness calculation

We provide these missing entities in a precalculated form for MSRC 21 dataset.
Please download it here: http://www.inf.ethz.ch/personal/vezhneva/Code/MSRC_data.zip
Also, graph construction is done in a simplified way:
- in-image spatial neighbours are not connected (set S is empty in eq.1);
- we do not restrict multi-image connections to superpixels that has less then 0.3 distance between each other;
- we use a Chi-square kernel instead of intersection (much faster).
This results in ~1% loss in per class accuracy.

MainScript.m implements a run of our MIM on the MSRC canonic train/test split.
CrossValidation.m implements cross-validation for the same dataset.

Feel free to contact me - alexander.vezhnevets@inf.ethz.ch