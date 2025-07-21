TITLE potassium leak channel
                                                                                
UNITS {
        (mV) = (millivolt)
        (mA) = (milliamp)
}
                                                                                
NEURON {
        SUFFIX Kleaksd
        USEION k READ ek WRITE ik
        RANGE gkl, ik
}
                                                                                
PARAMETER {
	v		(mV)
        gkl = .001        (mho/cm2)
        ek = -70      (mV)
}
                                                                                
ASSIGNED { ik    (mA/cm2)}
                                                                                
BREAKPOINT {
        ik = gkl*(v - ek)
}

