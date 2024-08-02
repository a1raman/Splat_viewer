from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO

# SH Constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = 1.0925484305920792
SH_C3 = 0.31539156525252005
SH_C4 = 0.5462742152960396

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
        
        # Calculate color using SH0, SH1, SH2 coefficients
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"] 
                    + SH_C1 * v["f_dc_3"] * v["x"]
                    + SH_C2 * v["f_dc_5"] * (2*v["z"]**2 - v["x"]**2 - v["y"]**2)
                    + SH_C3 * v["f_dc_6"] * (v["x"]**2 - v["y"]**2)
                    + SH_C4 * v["f_dc_8"] * (v["x"] * v["y"]),

                0.5 + SH_C0 * v["f_dc_1"]
                    + SH_C1 * v["f_dc_4"] * v["y"]
                    + SH_C2 * v["f_dc_5"] * (2*v["z"]**2 - v["x"]**2 - v["y"]**2)
                    + SH_C3 * v["f_dc_6"] * (v["x"]**2 - v["y"]**2)
                    + SH_C4 * v["f_dc_8"] * (v["x"] * v["y"]),

                0.5 + SH_C0 * v["f_dc_2"]
                    + SH_C1 * v["f_dc_4"] * v["z"]
                    + SH_C2 * v["f_dc_5"] * (2*v["z"]**2 - v["x"]**2 - v["y"]**2)
                    + SH_C3 * v["f_dc_6"] * (v["x"]**2 - v["y"]**2)
                    + SH_C4 * v["f_dc_8"] * (v["x"] * v["y"]),

                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()


def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)


def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.splat", help="The output SPLAT file."
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        splat_data = process_ply_to_splat(input_file)
        output_file = (
            args.output if len(args.input_files) == 1 else input_file + ".splat"
        )
        save_splat_file(splat_data, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
