# Custom builder for the SST cell

def SSTcellbuilder():
    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import Utils

    # Create the h object
    import os
    os.chdir("/data/SST/seg")
    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h

    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()

    return h