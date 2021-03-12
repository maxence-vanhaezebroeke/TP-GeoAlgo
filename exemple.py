import halfedge_mesh

# à mettre en paramètre lors de l'exec python
filename = "tests/data/cube.off"

# .off are supported
mesh = halfedge_mesh.HalfedgeMesh(filename)

mesh2 = halfedge_mesh.HalfedgeMesh("tests/data/mesh-02.off")
mesh3 = halfedge_mesh.HalfedgeMesh("tests/data/mesh-03.off")

cubeSmooth = halfedge_mesh.HalfedgeMesh("tests/data/cube-smooth.off")

# Returns a list of Vertex type (in order of file)--similarly for halfedges,
# and facets
#mesh.vertices

# The number of facets in the mesh
#print(len(mesh.facets))

#print(mesh.vertices[5])

''' TP3
print("halfedge puis faces du cube de base : ")
print(len(mesh.halfedges))
print(len(mesh.facets))
'''

# Get the halfedge that starts at vertex 25 and ends at vertex 50
#mesh.get_halfedge(25, 50)


#first_hedge = mesh.vertices[0].halfedge
#hedge = first_hedge.next_arround_edge()
#while hedge != first_hedge:
    #print(str(hedge) + " arround " + str(hedge.vertex.index))
    #hedge = hedge.next_arround_edge()


#mesh.dijkstra(mesh.vertices[0])

''' TP3
nb1, drapeau1 = mesh.composantes_connexes()
print("Nombre de composantes connexes du cube : ")
print(nb1)

nb2, drapeau2 = mesh2.composantes_connexes()
print("Nombre de composantes connexes du mesh-02 : ")
print(nb2)
print(drapeau2)

print("Calcul du genre de notre cube :")
print(mesh.calcul_genre())

print("Calcul du genre de mesh-02 : ")
print(mesh2.calcul_genre())
'''

###### TP4
''' Calcul propriété locale par moyenne
angles = mesh.calcul_angle_diedral_face_par_moyenne()
print(angles)
'''

#segmentation = cubeSmooth.segmentation_deux_classes("moyenne")
#print(segmentation)

#cubeSmooth.visualisation_segmentation("medianne")
#cubeSmooth.visualisation_propriete_locale()

print(cubeSmooth.visualisation_segmentation_composantes_connexes("mediane"))

        