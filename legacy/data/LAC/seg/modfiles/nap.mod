TITLE Sodium persistent current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX nap
	USEION na READ ena WRITE ina
	RANGE gbar, ina, minf, mtau, gna  :, vshift
}

PARAMETER { 
	gbar = 1e-4 	(mho/cm2)
	vshift = 0
	v ena 		(mV)  
} 
ASSIGNED { 
	ina 		(mA/cm2) 
	minf 		(1)
	mtau 		(ms) 
	gna		(mho/cm2)
} 
STATE {
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	gna = gbar * m
	ina = gna * ( v - ena ) 
} 

INITIAL { 
	settables(v-vshift) 
	m = minf
	:m = 0
} 

DERIVATIVE states { 
	settables(v-vshift) 
	m' = ( minf - m ) / mtau 
}
UNITSOFF
 
PROCEDURE settables(v) { 
	TABLE minf, mtau FROM -120 TO 40 WITH 641

	:Segregation
	if (v < -67.88) {
		minf = 0.009 * v + 0.624
	}
	if (v < -69.88) {
		minf = 0
	} else{
		minf  = 1 / ( 1 + exp( ( - v - 48 ) / 5 ) )
	}
	if( v < -40.0 ) {
		mtau = 600*(0.025 + 0.14 * exp( ( v + 40 ) / 10 ))
	}else{
		mtau = 600*(0.02 + 0.145 * exp( ( - v - 40 ) / 10 ))
	}
}
UNITSON
