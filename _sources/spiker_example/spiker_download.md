# Set up the model files

1. Go to your working directory.
2. Create a folder named `spiker`. This folder will contain `.hoc` and `.mod` files for the model.
3. Inside the `spiker` folder, create folders `orig` and `seg` (for the original and segmented model files respectively). Create the `modfiles` folder inside both the `orig` and `seg` folders.
4. For this example, we are using the default `hh.mod` modfile [provided by NEURON](https://github.com/neuronsimulator/nrn/blob/master/src/nrnoc/hh.mod). Download the modfile and put it inside each `modfiles` folder.
5. For this example, we are using the template [provided by NEURON](https://web.mit.edu/neuron_v7.4/nrntuthtml/tutorial/tutA.html).
    ```c
    begintemplate Spiker

    public soma, connect2target, type
    public all
    objref all

    create soma[1]
    public ncl, synlist
    objref ncl, synlist
    strdef type	


    proc init() {

        ncl = new List()
        synlist = new List()		
        type = "Spiker"
        
        all = new SectionList()
        for i=0,0 soma[0] {
            all.append()
        }

        soma[0] {
            nseg = 1  
            L = 18.8 
            diam = 18.8
        }
        
        soma[0] {
            insert hh_spiker
            Ra = 123.0

            // Target values
            cm = 1
            gl_hh_spiker = 0.0003
            el_hh_spiker = -54.3
            gnabar_hh_spiker = 0.12
            gkbar_hh_spiker = 0.036
        }
    }

    endtemplate Spiker
    ```

    Create the `template.hoc` file with the contents above. Put it inside both the `orig` and `seg` folders.

6. At this stage, you folder structure should look like this:
```
your_working_directory/
    spiker/
        orig/
            template.hoc
            modfiles/
                hh.mod
        seg/
            template.hoc
            modfiles/
                hh.mod
```