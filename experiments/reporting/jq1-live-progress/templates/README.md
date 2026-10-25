# Updating viewer

The standalone viewer displays all saved scalar losses and only actual retained
molecule frames. Controls distinguish global update numbers from retained-frame
indices, pause playback on manual input and clear absent matrix views. A latest
pointer permits refresh without editing previous immutable editions.

The refined presentation uses one primary loss for each experiment (contact,
affinity or the DBMol total), with a compact weighted DBMol breakdown and an
additional chemistry curve only when an active saved chemistry term exists.
It exposes no internal-metric selector. The larger raw-molecule panel uses
allocated-CPU bond-aware display coordinates. Every atom and bond, including
invalid/disconnected raw structures, remains intact. Matrices, full records and
provenance live in closed secondary disclosures.
