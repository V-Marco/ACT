# Custom builder for the L5 cell

def L5_orig_cell_builder():
    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import AllActiveUtils
    import os

    # Create the h object
    current_dir = os.getcwd()
    os.chdir("../../data/L5/orig")
    description = Config().load('manifest.json')
    utils = AllActiveUtils(description, axon_type = "stub")
    h = utils.h

    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    os.chdir(current_dir)

    return h

def L5_seg_cell_builder():
    from allensdk.model.biophys_sim.config import Config
    from allensdk.model.biophysical.utils import AllActiveUtils
    import os

    # Create the h object
    current_dir = os.getcwd()
    os.chdir("../../data/L5/seg_tuned")
    description = Config().load('manifest.json')
    utils = AllActiveUtils(description, axon_type = "stub")
    h = utils.h

    # Configure morphology
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))
    utils.load_cell_parameters()
    os.chdir(current_dir)

    return h