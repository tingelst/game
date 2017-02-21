import numpy as np
from pythreejs import *
from versor_pybind11 import *


def vector_mesh(vec, position=None, color='black', linewidth=2, arrow=False):
    length = vec.norm()
    line = Line(geometry=PlainGeometry(vertices=[[0, 0, 0], [0, length, 0]],
                                       colors=[color, color]),
                material=LineBasicMaterial(linewidth=linewidth,
                                           vertexColors='VertexColors'),
                type='LinePieces')
    if arrow:
        cone = Mesh(geometry=CylinderGeometry(radiusTop=0,
                                              radiusBottom=0.04,
                                              height=0.2,
                                              radiusSegments=20,
                                              heightSegments=1,
                                              openEnded=False),
                    material=LambertMaterial(color=color))
        cone.position = [0, length - 0.1, 0]
        line.children = [cone]
    line.quaternion = np.array(vec.unit().ratio(Vec(0, 1, 0)).quat()).tolist()
    if position is not None:
        line.position = np.array(position)[:3].tolist()
    return line


def bivector_mesh(biv, position=None, as_plane=False, color='gray'):

    weight = biv.norm()
    if as_plane:
        geometry = PlaneGeometry(width=weight, height=weight)
    else:
        geometry = CircleGeometry(radius=weight, segments=64)

    material = LambertMaterial(color=color)
    material.transparent = True
    material.opacity = 0.75
    mesh = Mesh(geometry=geometry, material=material)
    mesh.quaternion = np.array(biv.unit().duale().ratio(Vec(0, 0, 1)).quat(
    )).tolist()
    if position is not None:
        mesh.position = np.array(position)[:3].tolist()
    return mesh


def trivector_mesh(tri, color='gray'):
    geometry = BoxGeometry()
    material = LambertMaterial(color=color)
    mesh = Mesh(geometry=geometry, material=material)
    mesh.material.wireframe = True
    mesh.position = [0.5, 0.5, 0.5]

    return mesh


def sphere_mesh(sph, color='violet'):
    radius = sph.radius()
    geometry = SphereGeometry(radius=radius,
                              widthSegments=64,
                              heightSegments=64)
    material = LambertMaterial(color=color)
    material.transparent = True
    material.opacity = 0.75
    mesh = Mesh(geometry=geometry, material=material)
    mesh.position = np.array(sph.pnt())[:3].tolist()
    return mesh


def point_mesh(pnt, color='gray'):
    mesh = Mesh(geometry=SphereGeometry(radius=0.05,
                                        widthSegments=32,
                                        heightSegments=32),
                material=LambertMaterial(color=color))
    mesh.position = np.array(pnt)[:3].tolist()
    return mesh


def line_mesh(line, arrow=True, length=100, color='gray'):
    v = np.array(line.loc(Vec(0, 0, 0).null()))[:3]
    d = np.array(line.dir()) * 0.5 * length
    linesgeom = PlainGeometry(vertices=[(v + d).tolist(), (v - d).tolist()],
                              colors=[color, color])
    mesh = Line(geometry=linesgeom,
                material=LineBasicMaterial(linewidth=3,
                                           vertexColors='VertexColors'),
                type='LinePieces')
    if arrow:
        cone = Mesh(geometry=CylinderGeometry(radiusTop=0,
                                              radiusBottom=0.04,
                                              height=0.2,
                                              radiusSegments=20,
                                              heightSegments=1,
                                              openEnded=False),
                    material=LambertMaterial(color=color))
        cone.position = v.tolist()
        cone.quaternion = np.array(line.dir().vec().unit().ratio(Vec(
            0, 1, 0)).quat()).tolist()
        mesh.children = [cone]
    return mesh


def plane_mesh(plane, width=10, height=10, position=None, color='gray'):
    mesh = Mesh(geometry=PlaneGeometry(width=width, height=height),
                material=PhongMaterial(color=color,
                                         transparent=True,
                                         opacity=0.75))
    if type(plane) == Pln:
        normal = plane.dual().dir().vec()
        mesh.quaternion = np.array(normal.ratio(Vec(0, 0, 1)).quat()).tolist()
    else:
        normal = plane.dir().vec()
        mesh.quaternion = np.array(normal.ratio(Vec(0, 0, 1)).quat()).tolist()
    # mesh.children = [vector_mesh(Vec(0, 0, 1).spin(normal.ratio(Vec(0, 0, 1))))
    #                  ]
    if position is not None:
        mesh.position = np.array(position)[:3].tolist() 
    else:
        mesh.position = np.array(plane.loc(Vec(0, 0, 0).null()))[:3].tolist()

    return mesh


def circle_mesh(circle, color='gray'):
    mesh = Mesh(geometry=RingGeometry(innerRadius=circle.radius() - 0.01,
                                      outerRadius=circle.radius() + 0.01,
                                      thetaSegments=64),
                material=LambertMaterial(color=color),
                position=np.array(circle.pnt())[:3].tolist(),
                quaternion=circle.rot().quat().tolist())
    return mesh


def frame_mesh(position=None, quaternion=None, size=1, linewidth=2):
    frame = Line(geometry=PlainGeometry(
        vertices=[[0, 0, 0], [size, 0, 0], [0, 0, 0], [0, size, 0], [0, 0, 0],
                  [0, 0, size]],
        colors=['red', 'red', 'green', 'green', 'blue', 'blue']),
                 material=LineBasicMaterial(linewidth=linewidth,
                                            vertexColors='VertexColors'),
                 type='LinePieces')
    if position is not None:
        frame.position = position
    if quaternion is not None:
        frame.quaternion = quaternion
    return frame


class Colors:
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    HOTPINK = 'hotpink'
    LIGHTSKYBLUE = 'lightskyblue'
    AQUAMARINE = 'aquamarine'
    GRAY = 'gray'
    LIGHTGRAY = 'lightgray'
    BLACK = 'black'
    DEEPPINK = 'deeppink'
    DEEPSKYBLUE = 'deepskyblue'
