import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser
from typing import Dict, List, Optional

from smg.open3d import VisualisationUtil
from smg.utility import FiducialUtil, GeometryUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--fiducials_filename", "-f", type=str, default="fiducials-20210124T214842.txt",
        help="the name of the fiducials file"
    )
    parser.add_argument(
        "--gt_filename", "-g", type=str, default="spaint-20210124T214842_World.ply",
        help="the name of the ground-truth mesh file"
    )
    parser.add_argument(
        "--gt_render_style", type=str, choices=("hidden", "normal", "uniform"), default="normal",
        help="the rendering style to use for the ground-truth mesh"
    )
    parser.add_argument(
        "--input_filename", "-i", type=str, default="smglib-20210124T214842.ply",
        help="the name of the file containing the mesh to be evaluated"
    )
    parser.add_argument(
        "--input_render_style", type=str, choices=("hidden", "normal", "uniform"), default="uniform",
        help="the rendering style to use for the input mesh"
    )
    parser.add_argument(
        "--output_filename", "-o", type=str,
        help="the name of the file to which to save the transformed ground-truth mesh (if any)"
    )
    parser.add_argument(
        "--paint_uniform", "-p", action="store_true",
        help="whether to paint the meshes uniform colours to make it easier to compare them"
    )
    args: dict = vars(parser.parse_args())

    folder: str = "C:/spaint/build/bin/apps/spaintgui/meshes"
    fiducials_filename: str = os.path.join(folder, args["fiducials_filename"])
    gt_filename: str = os.path.join(folder, args["gt_filename"])
    input_filename: str = os.path.join(folder, args["input_filename"])
    output_filename: Optional[str] = os.path.join(folder, args["output_filename"]) \
        if args["output_filename"] is not None else None

    # Read in the mesh we want to evaluate, which should be metric and in world space.
    input_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(input_filename)

    # Load in the positions of the four marker corners as estimated during the ground-truth reconstruction.
    fiducials: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

    # Stack these positions into a 3x4 matrix.
    p: np.ndarray = np.column_stack([
        fiducials["0_0"],
        fiducials["0_1"],
        fiducials["0_2"],
        fiducials["0_3"]
    ])

    # Make another 3x4 matrix containing the world-space positions of the four marker corners.
    height: float = 1.5  # 1.5m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)

    q: np.ndarray = np.array([
        [-offset, -(height + offset), 0],
        [offset, -(height + offset), 0],
        [offset, -(height - offset), 0],
        [-offset, -(height - offset), 0]
    ]).transpose()

    # Estimate the rigid transformation between the two sets of points.
    transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)

    # Read in the ground-truth mesh, and transform it into world space using the estimated transformation.
    gt_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(gt_filename)
    gt_mesh = gt_mesh.transform(transform)

    # If requested, save the transformed ground-truth mesh to disk for later use.
    if output_filename is not None:
        # noinspection PyTypeChecker
        o3d.io.write_triangle_mesh(output_filename, gt_mesh)

    # Visualise the meshes to allow them to be compared.
    geometries: List[o3d.geometry.Geometry] = []
    if args["gt_render_style"] == "uniform":
        gt_mesh.paint_uniform_color((0.0, 1.0, 0.0))
    if args["gt_render_style"] != "hidden":
        geometries.append(gt_mesh)
    if args["input_render_style"] == "uniform":
        input_mesh.paint_uniform_color((1.0, 0.0, 0.0))
    if args["input_render_style"] != "hidden":
        geometries.append(input_mesh)
    VisualisationUtil.visualise_geometries(geometries)


if __name__ == "__main__":
    main()
