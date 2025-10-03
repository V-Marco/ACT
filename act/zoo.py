from neuron import h

# -------------
# Model classes
# -------------

class Model1C:

    L = 20
    diam = 20
    area = 3.14 * L * diam

    def __init__(self, channels: list):
        self.soma = [h.Section(name = "soma")]
        self.soma[0].nseg = 1
        self.soma[0].cm = 1 # (uF / cm2)
        self.soma[0].L = self.L # (um)
        self.soma[0].diam = self.diam # (um)

        self.all = [self.soma[0]]

        for channel in channels:
            print(channel)
            name, gbar = channel
            self.soma[0].insert(name)
            setattr(self.soma[0], f"gbar_{name}", gbar) # (S/cm2)

# --------
# Builders
# --------

def build_1c(channels):
    model = Model1C(channels)
    return model

def build_hh_spiker():
    channels = [("leak", 0.0003), ("Na", 0.12), ("K", 0.036)]
    model = Model1C(channels)
    return model

