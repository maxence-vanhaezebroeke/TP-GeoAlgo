import sys
from . import config
import math
import functools
import random

# python3 compatibility
try:
    xrange
except NameError:
    xrange = range
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()


# TODO: Reorder functions

class HalfedgeMesh:

    def __init__(self, filename=None, vertices=[], halfedges=[], facets=[]):
        """Make an empty halfedge mesh.

           filename   - a string that holds the directory location and name of
               the mesh
            vertices  - a list of Vertex types
            halfedges - a list of HalfEdge types
            facets    - a list of Facet types
        """

        self.vertices = vertices
        self.halfedges = halfedges
        self.facets = facets
        self.filename = filename
        # dictionary of all the edges given indexes
        # TODO: Figure out if I need halfedges or if I should just use edges
        # Which is faster?
        self.edges = None

        if filename:
            self.vertices, self.halfedges, self.facets, self.edges = \
                    self.read_file(filename)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and 
            (self.vertices, self.halfedges, self.facets) ==
            (other.vertices, other.halfedges, other.facets))

    def __hash__(self):
        return (hash(str(self.vertices)) ^ hash(str(self.halfedges)) ^ hash(str(self.facets)) ^ 
            hash((str(self.vertices), str(self.halfedges), str(self.facets))))

    def read_file(self, filename):
        """Determine the type of file and use the appropriate parser.

        Returns a HalfedgeMesh
#        """
        try:
            with open(filename, 'r') as file:

                first_line = file.readline().strip().upper()

                if first_line != "OFF":
                    raise ValueError("Filetype: " + first_line + " not accepted")

                # TODO: build OBJ, PLY parsers
                parser_dispatcher = {"OFF": self.parse_off}
                                      
                return parser_dispatcher[first_line](file)

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return
        except ValueError as e:
            print("Value error: {0}:".format(e))
            return

    def read_off_vertices(self, file_object, number_vertices):
        """Read each line of the file_object and return a list of Vertex types.
        The list will be as [V1, V2, ..., Vn] for n vertices

        Return a list of vertices.
        """
        vertices = []

        # Read all the vertices in
        for index in xrange(number_vertices):
            line = file_object.readline().split()

            try:
                # convert strings to floats
                line = list(map(float, line))
            except ValueError as e:
                raise ValueError("vertices " + str(e))

            vertices.append(Vertex(line[0], line[1], line[2], index))

        return vertices

    def parse_build_halfedge_off(self, file_object, number_facets, vertices):
        """Link to the code:
        http://stackoverflow.com/questions/15365471/initializing-half-edge-
        data-structure-from-vertices

        Pseudo code:
        map< pair<unsigned int, unsigned int>, HalfEdge* > Edges;

        for each face F
        {
            for each edge (u,v) of F
            {
                Edges[ pair(u,v) ] = new HalfEdge();
                Edges[ pair(u,v) ]->face = F;
            }
            for each edge (u,v) of F
            {
                set Edges[ pair(u,v) ]->nextHalfEdge to next half-edge in F
                if ( Edges.find( pair(v,u) ) != Edges.end() )
                {
                    Edges[ pair(u,v) ]->oppositeHalfEdge = Edges[ pair(v,u) ];
                    Edges[ pair(v,u) ]->oppositeHalfEdge = Edges[ pair(u,v) ];
            }
        }

        """
        Edges = {}
        facets = []
        halfedge_count = 0
        #TODO Check if vertex index out of bounds

        # For each facet
        for index in xrange(number_facets):
            line = file_object.readline().split()

            # convert strings to ints
            line = list(map(int, line))

            # TODO: make general to support non-triangular meshes
            # Facets vertices are in counter-clockwise order
            facet = Facet(line[1], line[2], line[3], index)
            facets.append(facet)

            # create pairing of vertices for example if the vertices are
            # verts = [1,2,3] then zip(verts, verts[1:]) = [(1,2),(2,3)]
            # note: we skip line[0] because it represents the number of vertices
            # in the facet.
            all_facet_edges = list(zip(line[1:], line[2:]))
            all_facet_edges.append((line[3], line[1]))

            # For every halfedge around the facet
            for i in xrange(3):
                Edges[all_facet_edges[i]] = Halfedge()
                Edges[all_facet_edges[i]].facet = facet
                Edges[all_facet_edges[i]].vertex = vertices[
                    all_facet_edges[i][1]]
                vertices[all_facet_edges[i][1]].halfedge = Edges[all_facet_edges[i]]
                halfedge_count +=1

            facet.halfedge = Edges[all_facet_edges[0]]

            for i in xrange(3):
                Edges[all_facet_edges[i]].next = Edges[
                    all_facet_edges[(i + 1) % 3]]
                Edges[all_facet_edges[i]].prev = Edges[
                    all_facet_edges[(i - 1) % 3]]

                # reverse edge ordering of vertex, e.g. (1,2)->(2,1)
                if all_facet_edges[i][2::-1] in Edges:
                    Edges[all_facet_edges[i]].opposite = \
                        Edges[all_facet_edges[i][2::-1]]

                    Edges[all_facet_edges[i][2::-1]].opposite = \
                        Edges[all_facet_edges[i]]

        return facets, Edges

    def parse_off(self, file_object):
        """Parses OFF files

        Returns a HalfedgeMesh
        """
        facets, halfedges, vertices = [], [], []

        # TODO Make ability to discard # lines
        vertices_faces_edges_counts = list(map(int, file_object.readline().split()))

        number_vertices = vertices_faces_edges_counts[0]
        vertices = self.read_off_vertices(file_object, number_vertices)

        number_facets = vertices_faces_edges_counts[1]
        facets, Edges = self.parse_build_halfedge_off(file_object,
                                                      number_facets, vertices)

        i = 0
        for key, value in iteritems(Edges):
            value.index = i
            halfedges.append(value)
            i += 1

        return vertices, halfedges, facets, Edges

    def get_halfedge(self, u, v):
        """Retrieve halfedge with starting vertex u and target vertex v

        u - starting vertex
        v - target vertex

        Returns a halfedge
        """
        return self.edges[(u, v)]
    
    ############################################# Creer fichier OFF avec couleur
    #Fonction ajoutée
    def write_off_mesh(self, filename):
        file = open(filename, "w")
        file.write("OFF\n")
        
        h = [len(self.vertices), len(self.facets), len(self.edges)]
        for p in h:
            file.write(str(p))
            file.write(" ")
            
        file.write("\n")
        for v in self.vertices:
            v.write_vertex(file)
        
        for face in self.facets:
            face.write_face(file)

        file.close()



    def update_vertices(self, vertices):
        # update vertices
        vlist = []
        i = 0
        for v in vertices:
            vlist.append(Vertex(v[0], v[1], v[2], i))
            i += 1
        self.vertices = vlist

        hlist = []
        # update all the halfedges
        for he in self.halfedges:
            vi = he.vertex.index
            hlist.append(Halfedge(None, None, None, self.vertices[vi], None,
                he.index))

        flist = []
        # update neighboring halfedges
        for f in self.facets:
            hi = f.halfedge.index
            flist.append(Facet(f.a, f.b, f.c, f.index,  hlist[hi]))
        self.facets = flist


        i = 0
        for he in self.halfedges:
            nextid = he.next.index
            oppid = he.opposite.index
            previd = he.prev.index

            hlist[i].next = hlist[nextid]
            hlist[i].opposite = hlist[oppid]
            hlist[i].prev = hlist[previd]


            fi = he.facet.index
            hlist[i].facet = flist[fi]
            i += 1

        self.halfedges = hlist


    def index_of_vertices(self):
        res = []
        for v in self.vertices:
            res.append(v.index)
        return res

    #####################################
    #Fonction ajoutée
    def dijkstra(self, source):
        Q = set()
        dists = {}
        prev = {}
        
        for v in self.vertices:            
            dists[v] = float("inf")
            prev[v] = None
            Q.add(v)
        dists[source] = 0

        while Q:
            u = min(Q, key = dists.get)
            #u = vertex in Q with min dists[u]
            Q.remove(u)
            
            for v in u.voisins(): # only v that are still in Q
                alt = dists[u] + u.distance(v)
                if (alt < dists[v]):
                    dists[v] = alt
                    prev[v] = u

        return dists, prev

    #Fonction ajoutée
    def parcours_voisins(self, v):
        h = v.halfedge
        first = True
        #print("on part du point " + str(v.index))

        distances = {}
        # parcours tous les voisins de v
        while first or h != v.halfedge:
            first = False
            v2 = h.opposite.vertex
            print(v2.index)

            distances[v2.index] = h.vertex.distance(v2)
            h = h.next_arround_vertex()
        
        return distances

    #Fonction ajoutée
    def closest(self,openVerts):
        mini = 99999
        for key,value in openVerts.items():
            if value < mini:
                mini = value

        return openVerts.get(mini)

    ### TP3 ##########################################

    #TODO: Je pense qu'il va falloir colorier les faces et aretes
    #drapeau doit retourner trois tableaux : un tableau pour les vertices
    #un tableau pour les halfedges et un tableau pour les facets

    #Fonction ajoutée
    def composantes_connexes(self):
        drapeau = [0 for v in self.vertices] #drapeau initiés à zéro
        cpt = 0

        #Pour chaque sommet, en partant du 1er sommet, on fait un parcours en profondeur
        #S'il reste des sommets non coloriés, ils appartiennent à une autre composante connexe
        couleur = 1
        for v in self.vertices:
            if drapeau[v.index] == 0:
                drapeau = self.parcours_en_profondeur(v, drapeau, couleur)
                cpt += 1
                couleur += 1

        return cpt, drapeau

    #Fonction ajoutée
    def parcours_en_profondeur(self, vertex, drapeau, couleur):
        #On colorie le sommet actuel (va servir pour le 1er sommet du 1er parcours)
        drapeau[vertex.index] = couleur
        h = vertex.halfedge
        h = h.next_arround_vertex()
        unvisited = []

        #On colorie tout les voisins de vertex
        while h != vertex.halfedge:
            voisin = h.opposite.vertex
            if drapeau[voisin.index] == 0:
                unvisited.append(voisin)
                drapeau[voisin.index] = couleur #Même pas obligatoire je pense

            h = h.next_arround_vertex()

        #Pour tous les points qui n'étaient pas visités, on en fait un parcours en prof (on continue)
        for v in unvisited:
            drapeau = self.parcours_en_profondeur(v, drapeau, couleur)

        return drapeau


    #Fonction privée : pas besoin de toucher
    def calcul_genre_une_composante(self):
        sommets = len(self.vertices)
        aretes = len(self.halfedges) / 2
        faces = len(self.facets)

        Euler = sommets - aretes + faces
        return int((2-Euler)/2)


    #TODO: si la fonction de composante connexe colorie bien les faces et demi aretes
    #On va pouvoir faire fonctionner cet algo

    #Fonction ajoutée
    #Retourne le calcul du genre
    def calcul_genre(self):
        nb, drapeau = self.composantes_connexes()
        if nb == 1:
            return self.calcul_genre_une_composante()

        listeG = []

        #c représente la couleur d'une composante connexe dans drapeau
        #ici, on itère pour chaque couleur (allant de 1 à nombre max de composante connexe)
        for c in range(1,max(drapeau)+1):
            #Pour chaque composante connexe, on va enregistrer cote, sommet, arete
            vertices = []
            halfedges = []
            facets = []

            #On recupere les points coloriés de la même couleur que la couleur actuelle
            for i in range(len(drapeau)):
                if drapeau[i] == c:
                    if self.vertices[i] not in vertices:
                        vertices.append(self.vertices[i])

            #FIXME: ici, ne fonctionne pas comme voulu
            #Il faut enregistrer dans "drapeau" la couleur des faces et demi-aretes
            #Et en fonction de cette couleur, on compte le nombre de faces et demi aretes
            for v in vertices:
                if v.halfedge not in halfedges:
                    halfedges.append(v.halfedge)
                if v.halfedge.facet not in facets:
                    facets.append(v.halfedge.facet)

            #Une fois qu'on a chopé sommets, faces et aretes, on fait le calcul
            print("Halfedge et facets d'une composante connexe : ")
            print(len(halfedges))
            print(len(facets))

            sommets = len(vertices)
            aretes = len(halfedges) / 2
            faces = len(facets)

            Euler = sommets - aretes + faces
            listeG.append(int((2-Euler)/2))
        
        return listeG


    ######################################### TP4

    #Fonction ajoutée
    def calcul_angle_diedral_face_par_moyenne(self):
        a = []
        res = {}
        for f in self.facets:
            a = f.get_every_angle_normal()
            moy = sum(a) / 3    #3 car 3 côtés d'un triangle
            res[f.index] = moy

        return res

    #Fonction ajoutée
    def visualisation_propriete_locale(self):
        angles = self.calcul_angle_diedral_face_par_moyenne()

        minA = min(angles.items(), key=lambda x : x[1])[1]
        maxA = max(angles.items(), key=lambda x : x[1])[1]
        
        for f in angles:
            degrade = self.get_color(minA, maxA, angles[f])
            self.facets[f].color = [255, degrade, degrade] 
        
        self.write_off_mesh("cube-visualisation.off")

    #Fonction ajoutée
    def get_color(self, min, max, angle):
        return int(((angle - min)/(max - min)) * 255)
    
    #Fonction ajoutée
    def segmentation_deux_classes(self,  methodeCalcul=None, seuil=None):
        angles = self.calcul_angle_diedral_face_par_moyenne()
        
        if methodeCalcul is None and seuil is None:
            return None

        if methodeCalcul == "moyenne":
            seuil = sum(angles.values()) / len(angles)
        
        elif methodeCalcul == "mediane":
            angles_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
            seuil = angles_sorted[int(len(angles_sorted) / 2)]
            
        elif methodeCalcul == "histogramme":
            pass

        res = {}

        for key,value in angles.items():
            if value < seuil:
                res[key] = 1
            else:
                res[key] = 2
        
        return res

    #Fonction ajoutée
    def visualisation_segmentation(self, methodeCalcul=None):
        if methodeCalcul is None:
            return

        segmentation = self.segmentation_deux_classes(methodeCalcul)

        red = [255, 0, 0]
        white = [255, 255, 255]
        for f in self.facets:
            if segmentation[f.index] == 1:
                f.color = red
            else:
                f.color = white

        self.write_off_mesh("cube-colore-segmentation.off")


    #Fonction ajoutée
    def visualisation_segmentation_composantes_connexes(self, methodeCalcul=None):
        if methodeCalcul is None:
            return

        segmentation = self.segmentation_deux_classes(methodeCalcul)

        red = [255, 0, 0]
        white = [255, 255, 255]

        colors = [[random.randint(0,255) for i in range(3)] for j in range(13000)]
        
        #print(segmentation)
        #segmentation[f.index] = sa couleur
        color = 2

        for k,v in segmentation.items():
            if v == 1:
                segmentation = self.colorie_face_connexe(segmentation, k, color)
                color += 1

        #print(segmentation)

        for k,v in segmentation.items():
            print(k,v)
            self.facets[k].color = colors[v]

        self.write_off_mesh("cube-colore-segmentation.off")

    #Fonction ajoutée
    def colorie_face_connexe(self, segmentation, index, color):
        segmentation[index] = color

        for f in self.facets[index].adjacent_faces_2():
            if segmentation[f.index] == 1:
                segmentation = self.colorie_face_connexe(segmentation, f.index, color)

        return segmentation


class Vertex:

    def __init__(self, x=0, y=0, z=0, index=None, halfedge=None):
        """Create a vertex with given index at given point.

        x        - x-coordinate of the point
        y        - y-coordinate of the point
        z        - z-coordinate of the point
        index    - integer index of this vertex
        halfedge - a halfedge that points to the vertex
        """

        self.x = x
        self.y = y
        self.z = z

        self.index = index

        self.halfedge = halfedge

    # pylint: disable=no-self-argument
    def __eq__(x, y):
        return x.__key() == y.__key() and type(x) == type(y)

    def __key(self):
        return (self.x, self.y, self.z, self.index)

    def __hash__(self):
        return hash(self.__key())

    def get_vertex(self):
        return [self.x, self.y, self.z]
    
    ##########################
    #Fonction ajoutée
    def write_vertex(self, file):
        c = [self.x, self.y, self.z]
        for p in c:
            file.write(str(p))
            file.write(" ")
        file.write("\n")

    #Fonction ajoutée
    def distance(self,v2):
        return math.sqrt((v2.x - self.x)**2 + (v2.y - self.y)**2 + (v2.z - self.z)**2)

    #Fonction ajoutée
    def voisins(self):
        out = []
        h = self.halfedge
        
        while True:
            v2 = h.opposite.vertex
            out.append(v2)
            
            h = h.next_around_vertex()
            if not (h != self.halfedge):
                break
        
        return out


class Facet:

    def __init__(self, a=-1, b=-1, c=-1, index=None, halfedge=None, color=[]):
        """Create a facet with the given index with three vertices.

        a, b, c - indices for the vertices in the facet, counter clockwise.
        index - index of facet in the mesh
        halfedge - a Halfedge that belongs to the facet
        
        color - color of the facet [R, G, B] from 0 to 255
        """
        self.a = a
        self.b = b
        self.c = c
        self.index = index
        # halfedge going ccw around this facet.
        self.halfedge = halfedge
        
        self.color = color

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c \
            and self.index == other.index and self.halfedge == other.halfedge

    def __hash__(self):
        return hash(self.halfedge) ^ hash(self.a) ^ hash(self.b) ^ \
            hash(self.c) ^ hash(self.index) ^ \
            hash((self.halfedge, self.a, self.b, self.c, self.index))
    
    ########################################
    #Fonction ajoutée
    def adjacent_faces(self):
        faces = []
        temp = self.halfedge
        faces.append(temp.opposite.facet)
        temp = self.halfedge.next_arround_vertex()
        while temp != self.halfedge:
            faces.append(temp.opposite.facet)
            temp = temp.next_arround_vertex()
        return faces

    def adjacent_faces_2(self):
        h = self.halfedge
        faces = []
        h2 = h
        while True:
            f2 = h.opposite.facet
            faces.append(f2)
            h2 = h2.next
            if h2 == h:
                break
        return faces
    
    ########################################

    #Fonction ajoutée
    def write_face(self, file):
        vertex = [self.halfedge.next.next.vertex.index, \
                self.halfedge.vertex.index, \
                self.halfedge.next.vertex.index]

        file.write(str(len(vertex)))
        file.write(" ")
        for v in vertex:
            file.write(str(v))
            file.write(" ")

        for p in self.color:
            file.write(str(p))
            file.write(" ")
        file.write("\n")

    def get_normal(self):
        """Calculate the normal of facet

        Return a python list that contains the normal
        """
        vertex_a = [self.halfedge.vertex.x, self.halfedge.vertex.y,
                    self.halfedge.vertex.z]

        vertex_b = [self.halfedge.next.vertex.x, self.halfedge.next.vertex.y,
                    self.halfedge.next.vertex.z]

        vertex_c = [self.halfedge.prev.vertex.x, self.halfedge.prev.vertex.y,
                    self.halfedge.prev.vertex.z]

        # create edge 1 with vector difference
        edge1 = [u - v for u, v in zip(vertex_b, vertex_a)]
        edge1 = normalize(edge1)
        # create edge 2 ...
        edge2 = [u - v for u, v in zip(vertex_c, vertex_b)]
        edge2 = normalize(edge2)

        # cross product
        normal = cross_product(edge1, edge2)

        normal = normalize(normal)

        return normal

    #Fonction ajoutée
    def get_every_angle_normal(self):
        current = self.halfedge
        next = self.halfedge.next
        res = []
        res.append(self.halfedge.get_angle_normal())
        while current != next:
            res.append(next.get_angle_normal())
            next = next.next

        return res



class Halfedge:

    def __init__(self, next=None, opposite=None, prev=None, vertex=None,
                 facet=None, index=None):
        """Create a halfedge with given index.
        """
        self.opposite = opposite
        self.next = next
        self.prev = prev
        self.vertex = vertex
        self.facet = facet
        self.index = index

    def __eq__(self, other):
        # TODO Test more
        return (self.vertex == other.vertex) and \
               (self.prev.vertex == other.prev.vertex) and \
               (self.index == other.index)

    def __hash__(self):
        return hash(self.opposite) ^ hash(self.next) ^ hash(self.prev) ^ \
                hash(self.vertex) ^ hash(self.facet) ^ hash(self.index) ^ \
                hash((self.opposite, self.next, self.prev, self.vertex,
                    self.facet, self.index))

    def get_angle_normal(self):
        """Calculate the angle between the normals that neighbor the edge.

        Return an angle in radians
        """
        a = self.facet.get_normal()
        b = self.opposite.facet.get_normal()

        dir = [self.vertex.x - self.prev.vertex.x,
               self.vertex.y - self.prev.vertex.y,
               self.vertex.z - self.prev.vertex.z]
        dir = normalize(dir)

        ab = dot(a, b)

        args = ab / (norm(a) * norm(b))

        if allclose(args, 1):
            args = 1
        elif allclose(args, -1):
            args = -1

        assert (args <= 1.0 and args >= -1.0)

        angle = math.acos(args)

        if not (angle % math.pi == 0):
            e = cross_product(a, b)
            e = normalize(e)

            vec = dir
            vec = normalize(vec)

            if (allclose(vec, e)):
                return angle
            else:
                return -angle
        else:
            return 0

    #Fonction ajoutée
    def next_arround_vertex(self):
        return self.opposite.prev


def allclose(v1, v2):
    """Compare if v1 and v2 are close

    v1, v2 - any numerical type or list/tuple of numerical types

    Return bool if vectors are close, up to some epsilon specified in config.py
    """

    v1 = make_iterable(v1)
    v2 = make_iterable(v2)

    elementwise_compare = list(map(
        (lambda x, y: abs(x - y) < config.EPSILON), v1, v2))
    return functools.reduce((lambda x, y: x and y), elementwise_compare)


def make_iterable(obj):
    """Check if obj is iterable, if not return an iterable with obj inside it.
    Otherwise just return obj.

    obj - any type

    Return an iterable
    """
    try:
        iter(obj)
    except:
        return [obj]
    else:
        return obj


def dot(v1, v2):
    """Dot product(inner product) of v1 and v2

    v1, v2 - python list

    Return v1 dot v2
    """
    elementwise_multiply = list(map((lambda x, y: x * y), v1, v2))
    return functools.reduce((lambda x, y: x + y), elementwise_multiply)


def norm(vec):
    """ Return the Euclidean norm of a 3d vector.

    vec - a 3d vector expressed as a list of 3 floats.
    """
    return math.sqrt(functools.reduce((lambda x, y: x + y * y), vec, 0.0))


def normalize(vec):
    """Normalize a vector

    vec - python list

    Return normalized vector
    """
    if norm(vec) < 1e-6:
        return [0 for i in xrange(len(vec))]
    return list(map(lambda x: x / norm(vec), vec))


def cross_product(v1, v2):
    """ Return the cross product of v1, v2.

    v1, v2 - 3d vector expressed as a list of 3 floats.
    """
    x3 = v1[1] * v2[2] - v2[1] * v1[2]
    y3 = -(v1[0] * v2[2] - v2[0] * v1[2])
    z3 = v1[0] * v2[1] - v2[0] * v1[1]
    return [x3, y3, z3]

def create_vector(p1, p2):
    """Contruct a vector going from p1 to p2.

    p1, p2 - python list wth coordinates [x,y,z].

    Return a list [x,y,z] for the coordinates of vector
    """
    return list(map((lambda x,y: x-y), p2, p1))
