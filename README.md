# SPIRAL v1.0
SPIRAL is a method for batch effect removing and coordinationg alignment for spatially resolved transcriptomes.  
![image](https://user-images.githubusercontent.com/17848453/183861675-b8b63081-83f3-4957-a3f8-0b822a4980cd.png)
# News
SPIRAL based on pyG (pytorch Geometric) framework is availabel at https://github.com/guott15/SPIRAL_pyG
# Usage
SPIRAL consistis two consective modules  
* **SPIRAL-integraion**    
 * Input  
   feature mat: Ncell x Ngene    
   edge mat: Nedge x 2    
   meta mat: Ncell x 1 or Ncell x 2: batch label (necessary) or domain label (selective)
* Output  
   embeddings: Ncell x zdim  
   recovery gene expression: Ncell x Ngene  
* **SPIRAL-alignemnt**  
 * Input    
   embeddings of SPIRAL-integration: Ncell x zdim  
   clusters  
 * Ouput  
   aligned coordinates: Ncell x 2  
* **Demo can be obtained in Demo floder**
# Installing
git clone https://github.com/guott15/SPIRAL.git  
cd SPIRAL  
python setup.py build  
python setup.py install  
#### Note: pytorch_revgrad should be in your current path.
# Data
data are stored at https://drive.google.com/drive/folders/1ujS3504u1Ge0wT3_LcCbGk_DNZut2SeK
