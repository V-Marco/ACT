This is to replicate Fig.4. in Alturki et al., 2016

AFTER YOU COMPILE THE FILES (see below):

ORIGINAL
To run the original model of Hemond et al:
1- Click on mosinit.hoc.
2- In the little poped-up window (first one from right), click:
	* Fig.9B for the Burst Firing cell
	* Fig.9C for the Adapting cell
	* Fig.9E for the Weakly Adapting cell

SEGREGATED
To run the segregated model of Hemond et al:
1- Click on mosinit.hoc.
2- In the little poped-up window (first one from right), click:
	* Fig.9B for the Burst Firing cell
	* Fig.9C for the Adapting cell
	* Fig.9E for the Weakly Adapting cell
    
____________________________________________________________________________
TO COMPILE AND RUN THE MODEL IN THE FOLDERS 'ORIGINAL' OR 'SEGREGATED' DO THE FOLLOWING:

Under unix systems:
to compile the mod files use the command 
nrnivmodl 
and run the simulation hoc file with the command 
nrngui FILENAME.hoc

Under Windows systems:
to compile the mod files use the "mknrndll" command.
A double click on the simulation file
FILENAME.hoc 
will open the simulation window.

Under MAC OS X:
Drag and drop the FILE folder onto the mknrndll icon in the NEURON
application folder. When the mod files are finished compiling, double click on the simulation file FILENAME.hoc
