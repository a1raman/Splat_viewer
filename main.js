//import { joystickfunc } from './joystick.js';
let cameras = [
    {
        id: 0,
        img_name: "00001",
        width: 2992,
        height: 2992,
        position: [
            0, 0, 0,
        ],
        rotation: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],

            // [-0.94212312, -0.04115569,  0.33273149],
            // [ 0.24994818,  0.57522362,  0.77887335],
            // [-0.22345008,  0.81696022, -0.53164468],
        ],
        fy: 923.04,
        fx: 919.12,
        // fy: 868.43888877239215,
        // fx: 862.50726029445627,
    }

];

let currentUrl = "4th_floor.splat";

// loadSplat 함수 추가
async function loadSplat(url) {
    currentUrl = url;
    await main();
}
const isSupported = !!(
	'ontouchstart' in window || // iOS & 안드로이드
    (navigator.pointerEanbled && navigator.maxTouchPoints > 0)
);  // IE 11+

let camera = cameras[0];

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}
// function translate4(a, x, y, z) {
//     return [
//         ...a.slice(0, 12),
//         a[0] * x + a[4] * y + a[8] * z + a[12],
//         a[1] * x + a[5] * y + a[9] * z + a[13],
//         a[2] * x + a[6] * y + a[10] * z + a[14],
//         a[3] * x + a[7] * y + a[11] * z + a[15],
//     ];
// }

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4_orbit(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

// 4x4 행렬 변환 함수
function translate4(m, tx, ty, tz) {
    let result = m.slice(); // 원본 행렬 복사
    result[12] += tx;
    result[13] += ty;
    result[14] += tz;
    return result;
}


// 벡터 변환 함수
function transformVector(matrix, vector) {
    let x = vector[0], y = vector[1], z = vector[2];
    return [
        matrix[0] * x + matrix[4] * y + matrix[8] * z,
        matrix[1] * x + matrix[5] * y + matrix[9] * z,
        matrix[2] * x + matrix[6] * y + matrix[10] * z
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

            // quaternions
            let scale = [
                f_buffer[8 * i + 3 + 0],
                f_buffer[8 * i + 3 + 1],
                f_buffer[8 * i + 3 + 2],
            ];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    function processPlyBuffer(inputBuffer) {
        const ubuf = new Uint8Array(inputBuffer);
        // 10KB ought to be enough for a header...
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);
        if (header_end_index < 0)
            throw new Error("Unable to read .ply file header");
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Vertex Count", vertexCount);
        let row_offset = 0,
            offsets = {},
            types = {};
        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32",
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };
        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        let dataView = new DataView(
            inputBuffer,
            header_end_index + header_end.length,
        );
        let row = 0;
        const attrs = new Proxy(
            {},
            {
                get(target, prop) {
                    if (!types[prop]) throw new Error(prop + " not found");
                    return dataView[types[prop]](
                        row * row_offset + offsets[prop],
                        true,
                    );
                },
            },
        );

        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            const size =
                Math.exp(attrs.scale_0) *
                Math.exp(attrs.scale_1) *
                Math.exp(attrs.scale_2);
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];

            const position = new Float32Array(buffer, j * rowLength, 3);
            const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3,
                4,
            );
            const rot = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3 + 4,
                4,
            );

            if (types["scale_0"]) {
                const qlen = Math.sqrt(
                    attrs.rot_0 ** 2 +
                        attrs.rot_1 ** 2 +
                        attrs.rot_2 ** 2 +
                        attrs.rot_3 ** 2,
                );

                rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
                rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
                rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
                rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                scales[0] = 0.01;
                scales[1] = 0.01;
                scales[2] = 0.01;

                rot[0] = 255;
                rot[1] = 0;
                rot[2] = 0;
                rot[3] = 0;
            }

            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            if (types["f_dc_0"]) {
                const SH_C0 = 0.28209479177387814;
                rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
                rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
                rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
            } else {
                rgba[0] = attrs.red;
                rgba[1] = attrs.green;
                rgba[2] = attrs.blue;
            }
            if (types["opacity"]) {
                rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
            } else {
                rgba[3] = 255;
            }
        }
        console.timeEnd("build buffer");
        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            postMessage({ buffer: buffer });
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {

    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis / viewport 
        + position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;
//precision mediump float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    //gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); // 예시로 흰색으로 설정
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}

`.trim();

let defaultViewMatrix = [
    1, 0, 0, 0, 
    0, 0, 1, 0, 
    0, -1, 0, 0, 
    0, 0, 0, 1,
];


let viewMatrix = defaultViewMatrix;
async function main() { 
    let carousel = false;
    const params = new URLSearchParams(location.search);
    
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    const url = new URL(
        currentUrl,
        "https://huggingface.co/spatialai/SplatViewer/resolve/main/"
    );
    const req = await fetch(url, {
        mode: "cors", // no-cors, *cors, same-origin
        credentials: "omit", // include, *same-origin, omit
    });
    console.log(req);
    if (req.status != 200)
        throw new Error(req.status + " Unable to load " + req.url);

    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const reader = req.body.getReader();
    let splatData = new Uint8Array(req.headers.get("content-length"));

    const downsample =
        splatData.length / rowLength > 500000 ? 1 : 1 / devicePixelRatio;
    console.log(splatData.length / rowLength, downsample);

    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));

        projectionMatrix = getProjectionMatrix(
            camera.fx,
            camera.fy,
            innerWidth,
            innerHeight,
        );

        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    worker.onmessage = (e) => {
        if (e.data.buffer) {
            splatData = new Uint8Array(e.data.buffer);
            const blob = new Blob([splatData.buffer], {
                type: "application/octet-stream",
            });
            const link = document.createElement("a");
            link.download = "model.splat";
            link.href = URL.createObjectURL(blob);
            document.body.appendChild(link);
            link.click();
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            // console.log(texdata)
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = [];
	let currentCameraIndex = 0;

    window.addEventListener("keydown", (e) => {
        // if (document.activeElement != document.body) return;
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key)
            camera = cameras[currentCameraIndex];
            viewMatrix = defaultViewMatrix;
            
            //viewMatrix = getViewMatrix(camera);
        }
		if (['-', '_'].includes(e.key)){
			currentCameraIndex = (currentCameraIndex + cameras.length - 1) % cameras.length;
			viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
		}
		if (['+', '='].includes(e.key)){
			currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
			viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
		}
        camid.innerText = "cam  " + currentCameraIndex;
        if (e.code == "KeyV") {
            location.hash =
                "#" +
                JSON.stringify(
                    viewMatrix.map((k) => Math.round(k * 100) / 100),
                );
                camid.innerText =""
        } else if (e.code === "KeyP") {
            // carousel = true;
            // camid.innerText =""
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

   
    let startX, startY, down = false;
    const sensitivity = 0.001; // 마우스 감도
    let accumulatedRotationX = 0; // 누적된 X축 회전 값
    let accumulatedRotationY = 0; // 누적된 Y축 회전 값
    
    // 초기 상태의 viewMatrix와 분리된 행렬들
    let positionMatrix = [
        1, 0, 0, 0, 
        0, 0, 1, 0, 
        0, -1, 0, 0, 
        0, 0, 0, 1
    ]; // 초기 위치 행렬
    
    let rotationMatrix = [
        1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 0, 1
    ]; // 초기 회전 행렬
    
    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = true;
    });
    
    canvas.addEventListener("mousemove", (e) => {
        if (!down) return;
    
        e.preventDefault();
    
        let dx = sensitivity * (e.clientX - startX);
        let dy = sensitivity * (e.clientY - startY);
    
        accumulatedRotationY += dx; // Y축 회전 값 누적 (왼쪽으로 이동하면 증가)
        accumulatedRotationX -= dy; // X축 회전 값 누적 (위로 이동하면 증가)
    
        // 회전 행렬 생성
        let rotationX = axisAngleRotationMatrix([1, 0, 0], accumulatedRotationX); // X축 회전
        let rotationY = axisAngleRotationMatrix([0, 1, 0], accumulatedRotationY); // Y축 회전
    
        // 회전 행렬을 결합하여 최종 rotationMatrix 생성
        rotationMatrix = multiplyMatrices(rotationY, rotationX);
    
        // 시작점 업데이트
        startX = e.clientX;
        startY = e.clientY;
    });
    
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
    });
    
    canvas.addEventListener("mouseout", (e) => {
        down = false;
    });
    
    // 4x4 행렬 곱셈 함수
    function multiplyMatrices(a, b) {
        let result = new Array(16).fill(0);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                for (let k = 0; k < 4; k++) {
                    result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                }
            }
        }
        return result;
    }
    
    // 축-각 회전 행렬 생성 함수
    function axisAngleRotationMatrix(axis, angle) {
        let [x, y, z] = axis;
        let c = Math.cos(angle);
        let s = Math.sin(angle);
        let t = 1 - c;
    
        return [
            t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
            0, 0, 0, 1
        ];
    }

    //touch orbit
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } 
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(rotationMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                //let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                //orbit 반경 (0:fps)
                let d = 0.3;
                inv = translate4_orbit(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                // inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4_orbit(inv, 0, 0, -d);

                rotationMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } 
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    //joystickfunc();
    //조이스틱 구현 부분
    const joystickMovement = document.getElementById('joystick-movement');
    const containerMovement = document.getElementById('joystick-container-movement');
    
    let touchIdMovement = null;
    //지속적 움직임을 위한 변수
    let moveX = 0, moveY = 0;
    let isMoving = false;

    containerMovement.addEventListener('touchstart', (event) => {
        if (touchIdMovement === null) {
            const touch = event.changedTouches[0];
            touchIdMovement = touch.identifier;
            isMoving = true;
            moveJoystickMovement(touch.clientX, touch.clientY);
            console.log("Touch down:", touch.clientX);
        }
    });

    containerMovement.addEventListener('touchmove', (event) => {
        if (touchIdMovement !== null) {
            const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdMovement);
            if (touch) {
                moveJoystickMovement(touch.clientX, touch.clientY);
            }
        }
    });

    containerMovement.addEventListener('touchend', (event) => {
        const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdMovement);
        if (touch) {
            touchIdMovement = null;
            isMoving = false;
            moveX = 0;
            moveY = 0;
            resetJoystickMovement();
        }
    });

    containerMovement.addEventListener('touchcancel', (event) => {
        const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdMovement);
        if (touch) {
            touchIdMovement = null;
            isMoving = false;
            moveX = 0;
            moveY = 0;
            resetJoystickMovement();
        }
    });

    function moveJoystickMovement(clientX, clientY) {
        const rect = containerMovement.getBoundingClientRect();
        const x = clientX - rect.left - rect.width / 2;
        const y = clientY - rect.top - rect.height / 2;
        const angle = Math.atan2(y, x);
        const distance = Math.min(Math.hypot(x, y), rect.width / 2 - joystickMovement.offsetWidth / 2);

        moveX = distance * Math.cos(angle);
        moveY = distance * Math.sin(angle);

        joystickMovement.style.transform = `translate(${moveX - 50}%, ${moveY - 50}%)`;

        // Update view matrix based on joystick movement
        updateViewMatrix(moveX, moveY);
    }

    function resetJoystickMovement() {
        joystickMovement.style.transform = 'translate(-50%, -50%)';
        updateViewMatrix(0, 0); // Reset view matrix when joystick is released
    }

    function updateViewMatrix(joystickX, joystickY) {
        // Normalize joystick inputs
        const normalize = (value, max) => value / max;
        const normalizedX = normalize(joystickX, containerMovement.clientWidth / 2);
        const normalizedY = normalize(joystickY, containerMovement.clientHeight / 2);

        const moveSpeed = 0.01;

        // Calculate movement vectors
        let rightVector = [rotationMatrix[0], rotationMatrix[1], rotationMatrix[2]];
        let forwardVector = [rotationMatrix[8], rotationMatrix[9], rotationMatrix[10]];

        // Normalize vectors
        const normalizeVector = (v) => {
            let length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return [v[0] / length, v[1] / length, v[2] / length];
        };

        rightVector = normalizeVector(rightVector);
        forwardVector = normalizeVector(forwardVector);

        let movement = [0, 0, 0];

        // Apply joystick movement for forward/backward movement
        movement[0] -= forwardVector[0] * normalizedY * moveSpeed;
        //movement[1] -= forwardVector[1] * normalizedY * moveSpeed;
        movement[2] += forwardVector[2] * normalizedY * moveSpeed;

        // Apply joystick movement for right/left movement
        movement[0] -= rightVector[0] * normalizedX * moveSpeed;
        //movement[1] -= rightVector[1] * normalizedX * moveSpeed;
        movement[2] += rightVector[2] * normalizedX * moveSpeed;

        // 계산된 이동을 positionMatrix에 적용
        positionMatrix = translate4(positionMatrix, movement[0], movement[1], movement[2]);

        // 이동된 positionMatrix와 회전된 rotationMatrix를 결합하여 viewMatrix 갱신
        viewMatrix = multiplyMatrices(positionMatrix, rotationMatrix);

        console.log('View Matrix:', viewMatrix); // For debugging purposes
    }
    //지속적인 움직임을 위한 함수
    function handleContinuousMovement() {
        if (isMoving) {
            // Update view matrix based on the last joystick movement
            updateViewMatrix(moveX, moveY);
        }
        requestAnimationFrame(handleContinuousMovement);
    }
    
    // 초기화 시 지속적인 이동 처리를 위한 함수 호출
    handleContinuousMovement();

    const joystickRotation = document.getElementById('joystick-rotation');
    const containerRotation = document.getElementById('joystick-container-rotation');

    let touchIdRotation = null;
    let startXRotation = 0;
    let startYRotation = 0;

    containerRotation.addEventListener('touchstart', (event) => {
        if (touchIdRotation === null) {
            const touch = event.changedTouches[0];
            touchIdRotation = touch.identifier;
            startXRotation = touch.clientX;
            startYRotation = touch.clientY;

        }
    });

    const maxRotationX = 10 * (Math.PI / 180); // 30도 라디안 상하 회전 제한
    const minRotationX = -10 * (Math.PI / 180); // -30도 라디안 상하 회전 제한

    containerRotation.addEventListener('touchmove', (event) => {
        if (touchIdRotation !== null) {
            const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdRotation);
            if (touch) {
                const dx = (touch.clientX - startXRotation) / containerRotation.clientWidth;
                const dy = (touch.clientY - startYRotation) / containerRotation.clientHeight;
                
                const rotateSpeed = 0.5;

                accumulatedRotationY += dx * rotateSpeed; // Y축 회전 값 누적 (왼쪽으로 이동하면 증가)
                accumulatedRotationX -= dy * rotateSpeed; // X축 회전 값 누적 (위로 이동하면 증가)

                // X축 회전 값 제한
                accumulatedRotationX = Math.max(minRotationX, Math.min(maxRotationX, accumulatedRotationX));

                // 회전 행렬 생성
                let rotationX = axisAngleRotationMatrix([1, 0, 0], accumulatedRotationX); // X축 회전
                let rotationY = axisAngleRotationMatrix([0, 1, 0], accumulatedRotationY); // Y축 회전
                rotationMatrix = multiplyMatrices(rotationY, rotationX);

                //rotateViewMatrix(dx);
                moveJoystickMovement2(touch.clientX, touch.clientY);
                startXRotation = touch.clientX;
                startYRotation = touch.clientY;
            }
        }
    });

    containerRotation.addEventListener('touchend', (event) => {
        const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdRotation);
        if (touch) {
            touchIdRotation = null;
            resetJoystick()
        }
    });

    

    function rotateY(matrix, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const rotationMatrix = [
            cos, 0, sin, 0,
            0, 1, 0, 0,
            -sin, 0, cos, 0,
            0, 0, 0, 1
        ];

        return multiplyMatrices(matrix, rotationMatrix);
    }
    function rotateX(matrix, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
    
        const rotationMatrix = [
            1, 0, 0, 0,
            0, cos, -sin, 0,
            0, sin, cos, 0,
            0, 0, 0, 1
        ];
    
        return multiplyMatrices(matrix, rotationMatrix);
    }

    
    function resetJoystick() {
        joystickRotation.style.transform = `translate(-50%, -50%)`;
    }

    function moveJoystickMovement2(clientX, clientY) {
        const rect = containerRotation.getBoundingClientRect();
        const x = clientX - rect.left - rect.width / 2;
        const y = clientY - rect.top - rect.height / 2;
        const angle = Math.atan2(y, x);
        const distance = Math.min(Math.hypot(x, y), rect.width / 2 - joystickRotation.offsetWidth / 2);

        const joystickX = distance * Math.cos(angle);
        const joystickY = distance * Math.sin(angle);

        joystickRotation.style.transform = `translate(${joystickX - 50}%, ${joystickY - 50}%)`;
    }
    //조이스틱 구현 끝
    
    let jumpDelta = 0;
    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    const frame = (now) => {
        
         // rotationMatrix에서 방향 벡터를 계산합니다.
        let rightVector = [rotationMatrix[0], rotationMatrix[1], rotationMatrix[2]];
        let upVector = [rotationMatrix[1], rotationMatrix[5], rotationMatrix[9]];
        let forwardVector = [rotationMatrix[2], rotationMatrix[6], rotationMatrix[10]];

        
        // 벡터 정규화 함수
        const normalize = (v) => {
            let length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return [v[0] / length, v[1] / length, v[2] / length];
        };

        // 벡터들을 정규화합니다
        rightVector = normalize(rightVector);
        upVector = normalize(upVector);
        forwardVector = normalize(forwardVector);
        
        const moveSpeed = 0.01;

        let movement = [0, 0, 0];

        if (activeKeys.includes("KeyW")) {
            movement[0] -= forwardVector[0] * moveSpeed;
            //movement[1] -= forwardVector[1] * moveSpeed;
            movement[2] -= forwardVector[2] * moveSpeed;
            
        }
        if (activeKeys.includes("KeyS")) {
            movement[0] += forwardVector[0] * moveSpeed;
            //movement[1] += forwardVector[1] * moveSpeed;
            movement[2] += forwardVector[2] * moveSpeed;
        }
        if (activeKeys.includes("KeyD")) {
            movement[0] -= rightVector[0] * moveSpeed;
            //movement[1] -= rightVector[1] * moveSpeed;
            movement[2] += rightVector[2] * moveSpeed;
        }
        if (activeKeys.includes("KeyA")) {
            movement[0] += rightVector[0] * moveSpeed;
            //movement[1] += rightVector[1] * moveSpeed;
            movement[2] -= rightVector[2] * moveSpeed;
        }
        
        // 계산된 이동을 positionMatrix에 적용
        positionMatrix = translate4(positionMatrix, movement[0], movement[1], movement[2]);

        // 이동된 positionMatrix와 회전된 rotationMatrix를 결합하여 viewMatrix 갱신
        viewMatrix = multiplyMatrices(positionMatrix, rotationMatrix);

        

        let inv = invert4(viewMatrix);


        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);

            const t = Math.sin((Date.now() - start) / 5000);
            inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotate4(inv, -0.6 * t, 0, 1, 0);

            viewMatrix = invert4(inv);
        }

       

        let inv2 = invert4(viewMatrix);
        inv2 = translate4(inv2, 0, -jumpDelta, 0);
        inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        let actualViewMatrix = invert4(inv2);
            
        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({ view: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }
        const progress = (100 * vertexCount) / (splatData.length / rowLength);
        if (progress < 100) {
            document.getElementById("progress").style.width = progress + "%";
        } else {
            document.getElementById("progress").style.display = "none";
        }
        fps.innerText = Math.round(avgFps) + " fps";
        if (isNaN(currentCameraIndex)){
            camid.innerText = "";
        }
        lastFrame = now;
        requestAnimationFrame(frame);
    };
        // 예시로 키보드 이벤트 처리
    
    // 초기 상태 설정
    let viewMatrix = multiplyMatrices(positionMatrix, rotationMatrix);
    //requestAnimationFrame(frame);
    //requestAnimationFrame(frame);

    frame();

    const selectFile = (file) => {
        const fr = new FileReader();
        if (/\.json$/i.test(file.name)) {
            fr.onload = () => {
                cameras = JSON.parse(fr.result);
                viewMatrix = getViewMatrix(cameras[0]);
                projectionMatrix = getProjectionMatrix(
                    camera.fx / downsample,
                    camera.fy / downsample,
                    canvas.width,
                    canvas.height,
                );
                gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

                console.log("Loaded Cameras");
            };
            fr.readAsText(file);
        } else {
            stopLoading = true;
            fr.onload = () => {
                splatData = new Uint8Array(fr.result);
                console.log("Loaded", Math.floor(splatData.length / rowLength));

                if (
                    splatData[0] == 112 &&
                    splatData[1] == 108 &&
                    splatData[2] == 121 &&
                    splatData[3] == 10
                ) {
                    // ply file magic header means it should be handled differently
                    worker.postMessage({ ply: splatData.buffer });
                } else {
                    worker.postMessage({
                        buffer: splatData.buffer,
                        vertexCount: Math.floor(splatData.length / rowLength),
                    });
                }
            };
            fr.readAsArrayBuffer(file);
        }
    };

    window.addEventListener("hashchange", (e) => {
        try {
            viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
            carousel = false;
        } catch (err) {}
    });

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("dragleave", preventDefault);
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectFile(e.dataTransfer.files[0]);
    });

    let bytesRead = 0;
    let lastVertexCount = -1;
    let stopLoading = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done || stopLoading) break;

        splatData.set(value, bytesRead);
        bytesRead += value.length;

        if (vertexCount > lastVertexCount) {
            worker.postMessage({
                buffer: splatData.buffer,
                vertexCount: Math.floor(bytesRead / rowLength),
            });
            lastVertexCount = vertexCount;
        }
    }
    if (!stopLoading)
        worker.postMessage({
            buffer: splatData.buffer,
            vertexCount: Math.floor(bytesRead / rowLength),
        });
}
// main 함수 호출 부분 수정
loadSplat(currentUrl).catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
