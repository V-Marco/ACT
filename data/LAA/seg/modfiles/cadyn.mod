:High-voltage activated Ca2+ channel

NEURON {
        SUFFIX cadyn
	USEION ca READ eca WRITE ica
	RANGE gcabar, gca,i
	RANGE uinf, zinf, utau, ztau 
}

UNITS {
        (mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gcabar = 0.0001 (mho/cm2) <0,1e9>
}

STATE { u z }

ASSIGNED {
	v (mV)
	eca (mV)
	ica (mA/cm2)
	i   (mA/cm2)
	uinf
	zinf 
	utau (ms)
	ztau (ms)
	gca (mho/cm2)
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	gca = gcabar*u*u*z
	ica = gca*(v-eca)
	i = ica
}

INITIAL {
	rate(v)
	u = uinf
	z = zinf
}

DERIVATIVE states {
	rate(v)
	u' = (uinf-u)/utau
	z' = (zinf-z)/ztau
}

PROCEDURE rate(v(mV)) {
	UNITSOFF
	uinf = 1/(exp(-(v+24.6)/11.3)+1)
	utau = 1.25*2/(exp(-0.031*(v+37.1)) + exp(0.031*(v+37.1)))

	:Segregation
	if (v < -38) {
	uinf = 0.116 * v + 4.646
	}
	if (v < -40) {
	uinf = 0
	}

	zinf = 1/(exp((v+12.6)/18.9)+1)
	ztau = 420	
	UNITSON	
}
