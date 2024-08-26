// Parameters
Lx = 0.020; // Length of the cuboid in the x direction
Ly = 0.007;  // Length of the cuboid in the y direction
Lz = 0.003;  // Length of the cuboid in the z direction
//lc = 0.000094; // Discretization step

strLC = Sprintf("%g", lc);


// Create the vertices of the cuboid
Point(1) = {0, 0, 0, lc};
Point(2) = {Lx, 0, 0, lc};
Point(3) = {Lx, Ly, 0, lc};
Point(4) = {0, Ly, 0, lc};
Point(5) = {0, 0, Lz, lc};
Point(6) = {Lx, 0, Lz, lc};
Point(7) = {Lx, Ly, Lz, lc};
Point(8) = {0, Ly, Lz, lc};

// Create the lines (edges) of the cuboid
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {1, 5};
Line(6) = {2, 6};
Line(7) = {3, 7};
Line(8) = {4, 8};
Line(9) = {5, 6};
Line(10) = {6, 7};
Line(11) = {7, 8};
Line(12) = {8, 5};

// Create the surfaces (faces) of the cuboid
Line Loop(1) = {1, 2, 3, 4}; //bottom
Line Loop(2) = {5, 9, -6, -1}; //left
Line Loop(3) = {6, 10, -7, -2}; //back
Line Loop(4) = {7, 11, -8, -3}; //right
Line Loop(5) = {9, 10, 11, 12}; //top
Line Loop(6) = {4, 5, -12, -8}; //front

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

// Create the volume of the cuboid
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Define physical groups for boundaries (surfaces)
//Physical Surface(0) = {1};
//Physical Surface(1) = {3};
//Physical Surface(2) = {5};
//Physical Surface(3) = {2};
//Physical Surface(4) = {4};
//Physical Surface(5) = {6};

// Define physical group for the volume
//Physical Volume(10) = {1};

// Mesh the volume
Mesh 3;

// Save the mesh to a file
//Mesh.Format = 1; // Set the file format (1 = MSH format)
Save StrCat("../meshes/cuboid-step-", strLC, ".msh");
