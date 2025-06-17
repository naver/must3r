# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import os
import torch
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    pass


def ply_read(ply_file):
    filedata = PlyData.read(ply_file)
    # Recover points 3D coordinates and colors from ply file
    return elements2colorpoints(filedata)


def elements2points(filedata):
    # Recover points 3D coordinates from elements (extracted from ply file)
    points = np.stack(filedata['vertex'])
    x_coords = points['x']
    y_coords = points['y']
    z_coords = points['z']

    # return 3D points format
    return np.transpose(np.squeeze(np.stack([x_coords, y_coords, z_coords])))


def elements2colorpoints(filedata):
    # Recover points 3D coordinates from elements (extracted from ply file)
    points = np.stack(filedata['vertex'])
    x_coords = points['x']
    y_coords = points['y']
    z_coords = points['z']

    try:
        r_colors = points['red']
    except:
        r_colors = 255 * np.ones_like(x_coords)

    try:
        g_colors = points['green']
    except:
        g_colors = 255 * np.ones_like(x_coords)

    try:
        b_colors = points['blue']
    except:
        b_colors = 255 * np.ones_like(x_coords)

    # return 3D points format
    return [np.transpose(np.squeeze(np.stack([x_coords, y_coords, z_coords]))), np.transpose(np.squeeze(np.stack([r_colors, g_colors, b_colors])))]


def elements2pointsfaces(filedata):
    # Recover points 3D coordinates from elements (extracted from ply file)
    points = np.stack(filedata['vertex'])
    x_coords = points['x']
    y_coords = points['y']
    z_coords = points['z']

    faces = np.stack(filedata['face']['vertex_indices'])

    # return 3D points format + faces (containing points indices)
    return [np.transpose(np.squeeze(np.stack([x_coords, y_coords, z_coords]))), faces]


def exportPointsPLY(vertices, outname="/tmp/test.ply"):  # color=[255,255,255],
    """ Export 3D point cloud in ply format. Input: [N,3]"""
    auto_create_f(outname)
    vertices = vertices.detach().cpu() if isinstance(vertices, torch.Tensor) else torch.tensor(vertices)
    B = vertices.shape[0]
    x, y, z = vertices.T.numpy()

    # connect the proper data structures
    vertices = np.empty(B, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')

    final_elements = PlyElement.describe(vertices, 'vertex')
    PlyData([final_elements]).write(outname)
    # reformatted_vs = [(r[0],r[1],r[2],0,0,-1.0,int(color[0]),int(color[1]),int(color[2])) for r in vertices]
    # desc_vertices = np.asarray(reformatted_vs,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),
    #                                                   ('nx', 'f4'), ('ny', 'f4'),('nz', 'f4'),
    #                                                   ('red', 'u1'),('green', 'u1'),('blue', 'u1')])
    # final_elements = PlyElement.describe(desc_vertices, 'vertex')
    # PlyData([final_elements]).write(outname)


def exportColoredPointsPLY(colored_vertices, outname="/tmp/test.ply", autocolors=False):
    """ Export 3D point cloud in ply format. Input: [N,6]"""

    auto_create_f(outname)

    if autocolors:
        if colored_vertices.shape[-1] != 3:
            print(f"Warning: AutoColor overwriting input colors in exportColoredPointsPLY(...) ")
        colored_vertices = autoselfcolor(colored_vertices)

    colored_vertices = colored_vertices.detach().cpu() if isinstance(
        colored_vertices, torch.Tensor) else torch.tensor(colored_vertices)
    B = colored_vertices.shape[0]
    x, y, z, red, green, blue = colored_vertices.T.numpy()

    # connect the proper data structures
    vertices = np.empty(B, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    final_elements = PlyElement.describe(vertices, 'vertex')
    PlyData([final_elements]).write(outname)

    # reformatted_vs = [(r[0],r[1],r[2],0,0,-1.0,int(r[3]),int(r[4]),int(r[5])) for r in colored_vertices]
    # desc_vertices = np.asarray(reformatted_vs,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),
    #                                                   ('nx', 'f4'), ('ny', 'f4'),('nz', 'f4'),
    #                                                   ('red', 'u1'),('green', 'u1'),('blue', 'u1')])
    # final_elements = PlyElement.describe(desc_vertices, 'vertex')
    # PlyData([final_elements]).write(outname)


def exportRaysPLY(in_rays, in_centers, outfile, n_interpol=50):  # vizualisation debug
    auto_create_f(outfile)
    if np.shape(in_rays) != np.shape(in_centers):
        print("Error, shape mismatch")
        return
    out_points = []
    for r_count, ray in enumerate(in_rays):
        for i in range(n_interpol):
            t_r = np.array(ray)
            c_r = np.array(in_centers[r_count])
            out_points.append(float(i / n_interpol) * t_r + float((n_interpol - i) / n_interpol) * c_r)
    exportPointsPLY(out_points, outfile)


def debugExportRays(rays, centers, name, interpols=20):
    auto_create_f(name)
    outpoints = [[0.0, 0.0, 0.0] for _ in range(interpols * np.shape(rays)[0])]
    for ii, ray in enumerate(rays):
        for interp in range(interpols):
            outpoints[ii * interpols + interp] = interp / interpols * centers[ii] + (1.0 - interp / interpols) * ray
    print("outpoints: ", np.shape(outpoints))
    exportPointsPLY(outpoints, name)


# Utils funcs
def autoselfcolor(verts):
    return np.concatenate([verts, verts.clip(0, 1) * 255], axis=-1)


def auto_create_f(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
