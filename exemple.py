import halfedge_mesh
import os

if not os.path.exists('out'):
    os.makedirs('out')

# mettre le nom du fichier .off en paramètre lors de l'exec python ?

cube = halfedge_mesh.HalfedgeMesh("tests/data/cube.off")

mesh2 = halfedge_mesh.HalfedgeMesh("tests/data/mesh-02.off")
mesh3 = halfedge_mesh.HalfedgeMesh("tests/data/mesh-03.off")

cubeSmooth = halfedge_mesh.HalfedgeMesh("tests/data/cube-smooth.off")

test1 = halfedge_mesh.HalfedgeMesh("tests/data/test1-2cc.off")

# Returns a list of Vertex type (in order of file)--similarly for halfedges,
# and facets
#cube.vertices

# The number of facets in the mesh
#print(len(cube.facets))

#print(cube.vertices[5])

''' TP3
print("halfedge puis faces du cube de base : ")
print(len(cube.halfedges))
print(len(cube.facets))
'''

# Get the halfedge that starts at vertex 25 and ends at vertex 50
#cube.get_halfedge(25, 50)


#first_hedge = cube.vertices[0].halfedge
#hedge = first_hedge.next_arround_edge()
#while hedge != first_hedge:
    #print(str(hedge) + " arround " + str(hedge.vertex.index))
    #hedge = hedge.next_arround_edge()


#cube.dijkstra(cube.vertices[0])

''' TP3
nb1, drapeau1 = cube.composantes_connexes()
print("Nombre de composantes connexes du cube : ")
print(nb1)

nb2, drapeau2 = mesh2.composantes_connexes()
print("Nombre de composantes connexes du mesh-02 : ")
print(nb2)
print(drapeau2)

print("Calcul du genre de notre cube :")
print(cube.calcul_genre())

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

#cubeSmooth.visualisation_segmentation("out/cube-colore-segmentation.off", methodeCalcul="mediane")
cubeSmooth.visualisation_propriete_locale("out/cube-visualisation.off")

cubeSmooth.visualisation_segmentation_composantes_connexes("out/cube-colore-segmentation.off", methodeCalcul="mediane")

test1.write_colored_CC("out/test1-2cc-colored.off")
mesh2.write_colored_CC("out/mesh02-cc-colored.off")
mesh3.write_colored_CC("out/mesh03-cc-colored.off")