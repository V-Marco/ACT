# 3D morphology builder for the L5 cell

def L5cellbuilder():
    from neuron import h
    # Load biophysics
    h.load_file("orig/L5PCbiophys3ActiveBasal.hoc")

    # Load morphology
    h.load_file("import3d.hoc")

    # Load template
    h.load_file("orig/L5PCtemplate.hoc")

    return getattr(h, "L5PCtemplate")("orig/cell1.asc")