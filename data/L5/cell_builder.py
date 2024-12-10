# 3D morphology builder for the L5 cell

def L5cellbuilder():
    from neuron import h
    # Load biophysics
    h.load_file("/Users/vladimiromelyusik/ACT/data/L5/orig/L5PCbiophys3ActiveBasal.hoc")

    # Load morphology
    h.load_file("import3d.hoc")

    # Load template
    h.load_file("/Users/vladimiromelyusik/ACT/data/L5/orig/L5PCtemplate.hoc")

    return getattr(h, "L5PCtemplate")("/Users/vladimiromelyusik/ACT/data/L5/orig/cell1.asc")