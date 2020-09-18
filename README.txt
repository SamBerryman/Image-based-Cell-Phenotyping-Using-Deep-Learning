This repository holds the software and protocols for the research article "Image-based Cell Phenotyping Using Deep Learning" 
later renamed "Image-based Phenotyping of Disaggregated Cells Using Deep Learning".

Preprint article: 

Image-based Cell Phenotyping Using Deep Learning Samuel Berryman, Kerryn Matthews, Jeong Hyun Lee, Simon P. 
Duffy, Hongshen Ma bioRxiv 817544; doi: https://doi.org/10.1101/817544

Software Contact:

Samuel Berryman
s.berryman@alumni.ubc.ca

Contents:

4-channel_model_8-classes.h5 - CNN model trained on 8 classes of cells
NISImagingProtocol.html - NIS-AR Imaging protocol for Nikon TI2-E
step1-Database.py - Localizing and extracting database of single cell images
step2-Training.py - Training CNN using the database files from step 1
step3-Testing.py  - Testing of the CNN, from step 2, on a seperate database collected using step 1

Data storgage: 

Scholars Portal Dataverse -> UBC Research Data Collection -> Fluorescent Microscopy Images of Disaggregated Cells
*Published link and doi to soon follow.
