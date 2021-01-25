import numpy as np
import open3d as o3d
import re

from argparse import ArgumentParser
from typing import Dict, List, Optional

from smg.open3d import VisualisationUtil
from smg.utility import GeometryUtil


def load_fiducials(filename: str) -> Dict[str, np.ndarray]:
    """
    Load named fiducials from a file.

    :param filename:    The name of the file containing the fiducials.
    :return:            A dictionary mapping fiducial names to positions in 3D space.
    """
    fiducials: Dict[str, np.ndarray] = {}

    vec_spec = r"\(\s*(.*?)\s+(.*?)\s+(.*?)\s*\)"
    line_spec = r".*?\s+(.*?)\s+" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + ".*"
    prog = re.compile(line_spec)

    def to_vec(x: int, y: int, z: int) -> np.ndarray:
        return np.array([float(m.group(x)), float(m.group(y)), float(m.group(z))])

    with open(filename, "r") as f:
        for line in f:
            m = prog.match(line)
            name = m.group(1)
            pos = to_vec(2, 3, 4)
            fiducials[name] = pos

    return fiducials


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--fiducials_filename", "-f", type=str,  # required=True,
        default="C:/spaint/build/bin/apps/spaintgui/meshes/fiducials-20210124T214842.txt",
        help="the name of the fiducials file"
    )
    parser.add_argument(
        "--gt_filename", "-g", type=str,  # required=True,
        default="C:/spaint/build/bin/apps/spaintgui/meshes/spaint-20210124T214842_World.ply",
        help="the name of the ground-truth mesh file"
    )
    parser.add_argument(
        "--gt_render_style", type=str, choices=("hidden", "normal", "uniform"), default="normal",
        help="the rendering style to use for the ground-truth mesh"
    )
    parser.add_argument(
        "--input_filename", "-i", type=str,  # required=True,
        default="C:/spaint/build/bin/apps/spaintgui/meshes/smglib.ply",
        help="the name of the file containing the mesh to be evaluated"
    )
    parser.add_argument(
        "--input_render_style", type=str, choices=("hidden", "normal", "uniform"), default="normal",
        help="the rendering style to use for the input mesh"
    )
    parser.add_argument(
        "--output_filename", "-o", type=str,
        default="C:/spaint/build/bin/apps/spaintgui/meshes/groundtruth.ply",
        help="the name of the file to which to save the transformed ground-truth mesh (if any)"
    )
    parser.add_argument(
        "--paint_uniform", "-p", action="store_true",
        help="whether to paint the meshes uniform colours to make it easier to compare them"
    )
    args: dict = vars(parser.parse_args())

    # Read in the mesh we want to evaluate, which should be metric and in world space.
    input_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(args["input_filename"])

    # Load in the positions of the four marker corners as estimated during the ground-truth reconstruction.
    fiducials: Dict[str, np.ndarray] = load_fiducials(args["fiducials_filename"])

    # Stack these positions into a 4x3 matrix.
    p: np.ndarray = np.column_stack([
        fiducials["0_0"],
        fiducials["0_1"],
        fiducials["0_2"],
        fiducials["0_3"]
    ])

    # Make another 4x3 matrix containing the world-space positions of the four marker corners.
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
    gt_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(args["gt_filename"])
    gt_mesh = gt_mesh.transform(transform)

    # If requested, save the transformed ground-truth mesh to disk for later use.
    output_filename: Optional[str] = args["output_filename"]
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
