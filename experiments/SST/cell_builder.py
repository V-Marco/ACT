# Custom builder for the SST cell

def sst_orig_cell_builder():
    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import Utils
    import os

    # Create the h object
    current_dir = os.getcwd()
    os.chdir("../../data/SST/orig")
    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h

    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    os.chdir(current_dir)

    return h

def sst_seg_cell_builder():
    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import Utils
    import os

    # Create the h object
    current_dir = os.getcwd()
    os.chdir("../../data/SST/seg")
    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h

    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    os.chdir(current_dir)

    return h