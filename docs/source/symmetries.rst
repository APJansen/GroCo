Symmetries
==========

The groups implemented in this package are listed below. 
We follow the crystallographic naming conventions.

The symmetries are combinations of translations along the grid axes, with rotations and reflections.
The latter form what is called a point group, because they always leave at least one point fixed.

The two types of symmetries can be combined in nontrivial ways, but since we are interested here in
only one image (or 3d volume) rather than an infinite periodic pattern, we can focus on the simple combination.

Furthermore because data typically comes on a square (or cubic) grid,
we restrict to the symmetries of a square (or cubic) grid (leaving aside the symmetries of a hexagonal or triangular grid).

2D: Wallpaper Groups
--------------------

The groups of symmetries of a 2D periodic pattern are called wallpaper groups.
For a detailed description of the wallpaper groups, see the `Wikipedia page <https://en.wikipedia.org/wiki/Wallpaper_group>`_.

After the restrictions above, the groups below remain.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - group
     - rotation
     - reflection
     - order
   * - P1
     - -
     - -
     - 1
   * - P2
     - 180°
     - -
     - 2
   * - PMw
     - -
     - y
     - 2
   * - PMh
     - -
     - x
     - 2
   * - P2MM
     - 180°
     - x, y
     - 4
   * - P4
     - 90°
     - -
     - 4
   * - P4MM
     - 90°
     - x, y
     - 8

Technically, PMw and PMh are not separate groups, they both correspond to the group PM with one
reflection. However, they are separated here for ease of use into two separate groups, the
PMw group with a reflection along the y-axis and the PMh group with a reflection along the x-axis.

3D: Space Groups
----------------

The groups of symmetries of a 3D periodic pattern are called space groups.

There are a lot more of them, for now we only implement the 4 largest groups.

.. list-table::
   :widths: 15 15 15 15 50
   :header-rows: 1

   * - group
     - rotation
     - reflection
     - order
     - description
   * - Oh
     - 90°
     - x, y, z
     - 48
     - all rotations and reflections of a cube
   * - O
     - 90°
     - -
     - 24
     - all rotations of a cube
   * - D4h
     - 90° around z axis
     - x, y, z
     - 16
     - All rotations and relfections of a cuboid (with depth different from width and height)
   * - D4
     - 90° around z axis
     - -
     - 8
     - All rotations of a cuboid (with depth different from width and height)
