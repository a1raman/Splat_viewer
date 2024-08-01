from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO

def process_ply_to_splat(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        
        # SH coefficients up to level 2 (9 coefficients in total)
        sh_coeffs = np.array([
            v["f_dc_0"], v["f_dc_1"], v["f_dc_2"],  # L0 (1 coefficient per color channel)
            v["f_rest_0"], v["f_rest_1"], v["f_rest_2"],  # L1 (3 coefficients)
            v["f_rest_3"], v["f_rest_4"], v["f_rest_5"],  # L2 (5 coefficients)
            v["f_rest_6"], v["f_rest_7"], v["f_rest_8"],
            v["f_rest_9"], v["f_rest_10"]
        ], dtype=np.float32)
        
        # Normalize SH coefficients
        SH_C0 = 0.28209479177387814
        SH_C1 = 0.48860251190291992
        SH_C2 = 1.0925484305920792
        
        sh_coeffs[0:3] *= SH_C0  # L0
        sh_coeffs[3:6] *= SH_C1  # L1
        sh_coeffs[6:11] *= SH_C2  # L2
        
        # Convert SH coefficients to RGB color (using only L0 for simplicity)
        color = np.array([
            0.5 + sh_coeffs[0],
            0.5 + sh_coeffs[1],
            0.5 + sh_coeffs[2],
            1 / (1 + np.exp(-v["opacity"])),
        ])
        
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )
        buffer.write(sh_coeffs.tobytes())
    
    return buffer.getvalue()

# ... (나머지 코드는 동일)
