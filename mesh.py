import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec

def load_data(filename, ext):
    """Function to load data from a file with the given extension."""
    file_path = f"{filename}.{ext}"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return np.loadtxt(file_path)

def case_reader(filename):
    faceMx = load_data(filename, 'fMx')
    ptCoordMx = load_data(filename, 'pMx')
    dia = load_data(filename, 'dia') if os.path.isfile(f"{filename}.dia") else np.ones(len(ptCoordMx))
    BC = load_data(filename, 'BC') if os.path.isfile(f"{filename}.BC") else np.array([[1, 1, 100], [100, 1, 0.1]])
    grpMx = load_data(filename, 'grpMx') if os.path.isfile(f"{filename}.grpMx") else np.empty((0,0))
    
    np_pts = len(ptCoordMx)
    nf = len(faceMx)
    nt = np_pts + nf
    
    return faceMx, ptCoordMx, grpMx, dia, BC, np_pts, nf, nt

def create_cylinder_segment(start_pt, end_pt, diameter, radial_divisions=10, length_divisions=10):
    start_pt = np.array(start_pt)
    end_pt = np.array(end_pt)
    
    length = np.linalg.norm(end_pt - start_pt)
    z = np.linspace(0, length, length_divisions)
    theta = np.linspace(0, 2 * np.pi, radial_divisions)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = (diameter / 2) * np.cos(theta_grid)
    y_grid = (diameter / 2) * np.sin(theta_grid)
    
    direction = (end_pt - start_pt) / length
    normal = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([1, 0, 0])
    normal /= np.linalg.norm(normal)
    binormal = np.cross(direction, normal)
    
    x_world = start_pt[0] + direction[0] * z_grid + x_grid * normal[0] + y_grid * binormal[0]
    y_world = start_pt[1] + direction[1] * z_grid + x_grid * normal[1] + y_grid * binormal[1]
    z_world = start_pt[2] + direction[2] * z_grid + x_grid * normal[2] + y_grid * binormal[2]
    
    return x_world, y_world, z_world

def export_to_vtp(filename, ptCoordMx, faceMx):
    points = vtk.vtkPoints()
    for pt in ptCoordMx:
        points.InsertNextPoint(pt)

    lines = vtk.vtkCellArray()
    for face in faceMx:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(face[1])-1)
        line.GetPointIds().SetId(1, int(face[2])-1)
        lines.InsertNextCell(line)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()

def plot_vascular_mesh(faceMx, ptCoordMx, dia):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in faceMx:
        start_idx, end_idx = int(face[1])-1, int(face[2])-1
        start_pt, end_pt = ptCoordMx[start_idx], ptCoordMx[end_idx]
        segment_dia = dia[start_idx]
        x_world, y_world, z_world = create_cylinder_segment(start_pt, end_pt, segment_dia)
        ax.plot_wireframe(x_world, y_world, z_world, color='black', linewidth=0.5)
        ax.scatter(start_pt[0], start_pt[1], start_pt[2], color='red')
        ax.scatter(end_pt[0], end_pt[1], end_pt[2], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Vascular Mesh')
    plt.show()

def export_to_step(filename, ptCoordMx, faceMx, dia):
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for face in faceMx:
        start_idx, end_idx = int(face[1]) - 1, int(face[2]) - 1
        start_pt = gp_Pnt(*ptCoordMx[start_idx])
        end_pt = gp_Pnt(*ptCoordMx[end_idx])
        
        direction = gp_Dir(gp_Vec(gp_Pnt(*start_pt), gp_Pnt(*end_pt)))
        edge = BRepBuilderAPI_MakeEdge(start_pt, end_pt).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        
        face = BRepBuilderAPI_MakeFace(wire).Face()
        prism = BRepPrimAPI_MakePrism(face, gp_Vec(direction.X() * dia[start_idx], direction.Y() * dia[start_idx], direction.Z() * dia[start_idx])).Shape()
        
        builder.Add(compound, prism)

    step_writer = STEPControl_Writer()
    step_writer.Transfer(compound, STEPControl_AsIs)
    step_writer.Write(filename)

filename = 'cow.cs31'  
try:
    faceMx, ptCoordMx, grpMx, dia, BC, np_pts, nf, nt = case_reader(filename)
    plot_vascular_mesh(faceMx, ptCoordMx, dia)
    export_to_vtp("vascular_mesh.vtp", ptCoordMx, faceMx)
    export_to_step("vascular_mesh.stp", ptCoordMx, faceMx, dia)
except FileNotFoundError as e:
    print(e)
