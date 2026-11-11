"""Groups Mosaic's small-molecule design code into one Python package.

Start with generate_molecule.py to see how a run is selected. The Stage0
folder contains the first design loop, and core provides the shared pieces
it uses: preparing the protein and ligand, scoring contacts, and updating
the molecule. Importing this package does not start a design.
"""
