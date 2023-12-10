: alf-dendrotoxin sensitive, slowly inactivating channel

NEURON {
	SUFFIX kdtx
	USEION k READ ek WRITE ik
	RANGE gkdtxbar, gkdtx
	RANGE uinf, zinf, utau, ztau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gkdtxbar = 0.001 (siemens/cm2) <0,1e9>
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	uinf
	zinf 
	utau (ms)
	ztau (ms)
	gkdtx (siemens/cm2)
}

STATE {
	u z
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gkdtx = gkdtxbar*u*z
	ik = gkdtx*(v-ek)
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
	uinf = 1/(exp(-(v+8.6)/11.1)+1)
	utau = 1.5
	zinf = 1/(exp((v+21)/9)+1)
	ztau = 569
	UNITSON
}