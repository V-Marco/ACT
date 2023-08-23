"""
The purpose of this module is to assist in the creation of single cells, from scratch
"""

cell_hoc_template = """
{{load_file("stdrun.hoc")}}
{{load_file("nrngui.hoc")}}

begintemplate {cell_name}

	public soma
	create soma[1]

	public all, somatic
    objref all, somatic

	proc init() {{	
		all = new SectionList()
		somatic = new SectionList()
		soma[0] all.append()
		soma[0] somatic.append()

		{global_params}	
		
		/////// topology ////////
		// No connections to be made
		
		/////// geometry ////////
		soma[0] {{
            {geometry_section}
        }}

        //////// mechanisms /////////
		soma[0] {{	
			{mechanism_section}
		}}
				
	define_shape()// builtin fcn: fill in 3d info for sections defined
	}}//end init

endtemplate {cell_name}
"""


class CellGenerator:

    """
    Generate a cell

    eg:

    from act.builder import CellGenerator

    pyramidal_cell = CellGenerator('pyr_type_A')
    pyramidal_cell.set_global_params(Rm = 80000, Cm = 2.4, Ra = 150)
    pyramidal_cell.insert_mech('leak', el_leak = -72)
    pyramidal_cell.set_geometry(L = 25, diam = 25, nseg = 1)
    pyramidal_cell.write_hoc('cell_templates.hoc')

    """

    def __init__(self, cell_name):
        self.cell_name = cell_name
        self.global_params = {}
        self.mods = {}
        self.set_geometry()

    def set_geometry(self, L=25, diam=25, nseg=1):
        """
        Set geometric properties for the cell
        """
        self.L = L
        self.diam = diam
        self.nseg = nseg

    def set_global_params(self, *args, **kwargs):
        """
        Set global properties like Ra and Cm for the cell
        """
        for key, value in kwargs.items():
            self.global_params[key] = value

    def insert_mech(self, mechanism, **params):
        """
        Insert a new mechanism and properties, can update
        """
        self.mods[mechanism] = params

    def write_hoc(self, file_path, append=False):
        """
        Take the cell that has been built and write to a file_path

        params:
        file_path: where the files is written to
        append: if you have multiple templates that you're writting to the same file, set to True
        """

        mechanism_section = ""
        for mech, params in self.mods.items():
            mech_line = f"insert {mech} "
            for param, value in params.items():
                mech_line += f"{param} = {value} "
            mechanism_section = mechanism_section + mech_line + "\n"

        global_params = ""
        for param, value in self.global_params.items():
            global_params += f"{param} = {value}\n"

        geometry_section = f"""
        L = {self.L}
        diam = {self.diam}
        nseg = {self.nseg}
        """

        cell_hoc = cell_hoc_template.format(
            cell_name=self.cell_name,
            global_params=global_params,
            geometry_section=geometry_section,
            mechanism_section=mechanism_section,
        )

        # write to file
        with open(file_path, "a" if append else "w") as text_file:
            text_file.write(cell_hoc)
