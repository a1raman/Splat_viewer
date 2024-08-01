let cameras = [ //focal용
    {
        id: 0,
        img_name: "00001",
        width: 3840,
        height: 2160,
        position: [
            -0.1910679031899027, -0.00010180568736527842, -0.874154583663699,
        ],
        rotation: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        fy: 923.04,
        fx: 919.12,
    }
];
//변수들
let camera = cameras[0];
// 전역 변수로 캐시할 객체 선언
let cachedData = null;
let noDataCheck = false; //clusterData 없을 때,
let moveRangeX = [];
let moveRangeY = [];
let moveRangeZ = [];
let polygonVertices = [];
let convexHull = [];
//Unity의 Start함수같은 것
let positionMatrix = [
    // 1,0,0,0,
    // 0,1,0,0,
    // 0,0,1,0,
    // 0,0,0,1
    1, 0,  0, 0,
    0, 0, -1, 0,
    0, 1,  0, 0,
    0, 0,  0, 1
];

let defaultViewMatrix = [
    1,0,0,0,
    0,0,1,0,
    0,-1,0,0,
    0,0,0,1,
    // 1,0,0,0,
    // 0,1,0,0,
    // 0,0,1,0,
    // 0,0,0,1,
];
let viewMatrix = defaultViewMatrix;

// Vector3 클래스 정의
class Vector3 {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    static fromArray(array) {
        return new Vector3(array[0], array[1], array[2]);
    }

    static subtract(a, b) {
        return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    static cross(a, b) {
        return new Vector3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static dot(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static normalize(v) {
        const length = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (length === 0) {
            return new Vector3(0, 0, 0);
        }
        return new Vector3(v.x / length, v.y / length, v.z / length);
    }

    static scale(v, s) {
        return new Vector3(v.x * s, v.y * s, v.z * s);
    }

    static add(a, b) {
        return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
}
function subtract(vecA, vecB) {
    return [vecA[0] - vecB[0], vecA[1] - vecB[1], vecA[2] - vecB[2]];
}
function cross(vecA, vecB) {
    return [
        vecA[1] * vecB[2] - vecA[2] * vecB[1],
        vecA[2] * vecB[0] - vecA[0] * vecB[2],
        vecA[0] * vecB[1] - vecA[1] * vecB[0]
    ];
}

function dot(vecA, vecB) {
    return vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2];
}
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
// 3x3
function multiply3x3Matrices(a, b) {
    let result = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            for (let k = 0; k < 3; k++) {
                result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    return result;
}
function convert3x3To4x4(matrix3x3) {
    if (matrix3x3.length !== 9) {
        throw new Error("Input matrix must have 9 elements.");
    }

    let matrix4x4 = [
        matrix3x3[0], matrix3x3[1], matrix3x3[2], 0,
        matrix3x3[3], matrix3x3[4], matrix3x3[5], 0,
        matrix3x3[6], matrix3x3[7], matrix3x3[8], 0,
        0, 0, 0, 1
    ];

    return matrix4x4;
}

// 축-각 회전 행렬 생성 함수
function axisAngleRotationMatrix(axis, angle) {
    let rad = angle * Math.PI / 180;
    let cos = Math.cos(rad);
    let sin = Math.sin(rad);
    let [x, y, z] = axis;
    let t = 1 - cos;

    return [
        t * x * x + cos,       t * x * y - z * sin,   t * x * z + y * sin,
        t * x * y + z * sin,   t * y * y + cos,       t * y * z - x * sin,
        t * x * z - y * sin,   t * y * z + x * sin,   t * z * z + cos
    ];
}

// 카메라의 뷰 행렬을 업데이트하는 함수
function updateViewMatrix() {
    const eye = [positionMatrix[12], positionMatrix[13], positionMatrix[14]];
    const target = [0, -1, 0]; // 여기서는 원점(0, 0, 0)을 바라보도록 설정
    const up = [0, 0, -1]; // Y축을 위쪽 방향으로 설정

    const zAxis = normalize(subtract(eye, target));
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    // 뷰 행렬 계산
    viewMatrix = [
        xAxis[0], yAxis[0], zAxis[0], 0,
        xAxis[1], yAxis[1], zAxis[1], 0,
        xAxis[2], yAxis[2], zAxis[2], 0,
        -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1
    ];
    return viewMatrix;
}
function createIdentityMatrix3x3() {
    return {
        value: [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ]
    };
}

function getOrientation(p, q, r) {
    let val = (q.z - p.z) * (r.x - q.x) - (q.x - p.x) * (r.z - q.z);
    if (val === 0) return 0;  // collinear
    return (val > 0) ? 1 : 2; // clock or counterclockwise
}

function distanceSquared(p, q) {
    return (q.x - p.x) * (q.x - p.x) + (q.z - p.z) * (q.z - p.z);
}

function nextToTop(stack) {
    let top = stack.pop();
    let nextToTop = stack[stack.length - 1];
    stack.push(top);
    return nextToTop;
}
// 벡터 정규화 함수
function normalize(v) {
    let length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}
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

function rotate4(m, angle, x, y, z) {
    // 각도를 라디안으로 변환
    const rad = angle * Math.PI / 180;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const t = 1 - c;

    // 축 벡터 정규화
    const length = Math.sqrt(x * x + y * y + z * z);
    x /= length;
    y /= length;
    z /= length;

    // 회전 행렬 생성
    const r = [
        t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
        t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
        0, 0, 0, 1
    ];

    // 행렬 곱셈: r * m
    return [
        r[0] * m[0] + r[1] * m[4] + r[2] * m[8] + r[3] * m[12],
        r[0] * m[1] + r[1] * m[5] + r[2] * m[9] + r[3] * m[13],
        r[0] * m[2] + r[1] * m[6] + r[2] * m[10] + r[3] * m[14],
        r[0] * m[3] + r[1] * m[7] + r[2] * m[11] + r[3] * m[15],

        r[4] * m[0] + r[5] * m[4] + r[6] * m[8] + r[7] * m[12],
        r[4] * m[1] + r[5] * m[5] + r[6] * m[9] + r[7] * m[13],
        r[4] * m[2] + r[5] * m[6] + r[6] * m[10] + r[7] * m[14],
        r[4] * m[3] + r[5] * m[7] + r[6] * m[11] + r[7] * m[15],

        r[8] * m[0] + r[9] * m[4] + r[10] * m[8] + r[11] * m[12],
        r[8] * m[1] + r[9] * m[5] + r[10] * m[9] + r[11] * m[13],
        r[8] * m[2] + r[9] * m[6] + r[10] * m[10] + r[11] * m[14],
        r[8] * m[3] + r[9] * m[7] + r[10] * m[11] + r[11] * m[15],

        r[12] * m[0] + r[13] * m[4] + r[14] * m[8] + r[15] * m[12],
        r[12] * m[1] + r[13] * m[5] + r[14] * m[9] + r[15] * m[13],
        r[12] * m[2] + r[13] * m[6] + r[14] * m[10] + r[15] * m[14],
        r[12] * m[3] + r[13] * m[7] + r[14] * m[11] + r[15] * m[15]
    ];
}
//시작함수, clusterdata fetch
document.addEventListener('DOMContentLoaded', (event) => {
    const errorMessageElement = document.getElementById('error-message');

    function fetchAndProcessImageText(callback) {
        fetch('https://huggingface.co/spatialai/SplatViewer/resolve/main/DWcluster_data.txt')
            .then(response => {
                if (!response.ok) {
                    dataCheck = false;
                    throw new Error(`Cluster 파일이 없거나, 네트워크 오류입니다. 원점에서 움직일 수 없습니다.`);
                }
                return response.text();
            })
            .then(text => {
                const lines = text.trim().split('\n');

    
                for (let i = 0; i < lines.length; i++) {
                    const parts = lines[i].trim().split(/\s+/); // 공백 또는 여러 개의 공백 문자를 구분자로 사용
    
                    const x = parseFloat(parts[0]);
                    const y = parseFloat(parts[2]); //모델좌표가 달라 다르게 fetch
                    const z = parseFloat(parts[1]);
    
                    // 직접 정의한 Vector3를 사용하여 벡터 생성
                    const vertex = new Vector3(x, y*-1, z);
                    //console.log(vertex);
                    // 새로운 Vector3 생성하여 리스트에 추가
                    polygonVertices.push(vertex);
                    //console.log(polygonVertices);
    
                }
                if (typeof callback === 'function') {
                    callback(polygonVertices);
                } else {
                    console.error('콜백 함수가 올바르지 않습니다.');
                }
            })
            .catch(error => {
                console.error('파일을 가져오는 도중 오류가 발생했습니다:', error);
                errorMessageElement.textContent = error.message;
                errorMessageElement.style.display = 'block'; // 오류 메시지를 표시


            });
    }
    fetchAndProcessImageText(polygonVertices => {
        console.log('파싱된 데이터:', polygonVertices);

        convexHull = ComputeConvexHull(polygonVertices);
        console.log('계산된 볼록 껍질:', convexHull);
        polygonVertices.length = 0; // 배열 비우기
        polygonVertices.push(...convexHull);

        console.log('계산된 볼록 껍질 잘 넣었니?:', polygonVertices);

        console.log('계산된 볼록 껍질의 길이:', polygonVertices.length);
    });
    

});

//cluster data들이 다각형 안에 있는지 확인
function IsPointInsidePolygon(point) {
    let vertexCount = polygonVertices.length;
    let inside = false;

    // 다각형 정점들의 y와 z 값을 교환 (좌표계가 다를 경우 교환)
    const transformedVertices = polygonVertices.map(vertex => {
        return new Vector3(vertex.x, vertex.y, vertex.z);
    });

    // 입력 점의 y와 z 값을 교환 (좌표계가 다를 경우 교환)
    const transformedPoint = new Vector3(point.x, point.y, point.z);
    
    // 교차 여부 검사
    for (let i = 0, j = vertexCount - 1; i < vertexCount; j = i++) {
        let vertex1 = transformedVertices[i];
        let vertex2 = transformedVertices[j];

        // 교차점이 있는지 확인
        if (((vertex1.z > transformedPoint.z) !== (vertex2.z > transformedPoint.z)) &&
            (transformedPoint.x < (vertex2.x - vertex1.x) * (transformedPoint.z - vertex1.z) / (vertex2.z - vertex1.z) + vertex1.x)) {
            inside = !inside;
        }
    }

    return inside;
}
//ConvexHull 알고리즘 구현
function ComputeConvexHull(points) {
    if (points.length <= 1) return points;

    // Y 축을 기준으로 가장 아래쪽에 있는 점을 찾음
    let pivot = points.reduce((acc, point) => {
        if (point.z < acc.z || (point.z === acc.z && point.x < acc.x)) {
            return point;
        }
        return acc;
    }, points[0]);

    points = points.filter(point => point !== pivot);
    console.log("시작점",pivot);
    // 피벗 점을 기준으로 시계 방향 정렬
    points.sort((a, b) => {
        let angleA = Math.atan2(a.z - pivot.z, a.x - pivot.x);
        let angleB = Math.atan2(b.z - pivot.z, b.x - pivot.x);
        return angleA - angleB;
    });
    //console.log(points);
    // 그레이엄 스캔 알고리즘을 이용한 볼록 껍질 계산
    let hull = [];
    hull.push(pivot);
    hull.push(points[0]);

    for (let i = 1; i < points.length; i++) {
        let top = hull.pop();
        while (hull.length > 0 && getOrientation(hull[hull.length - 1], top, points[i]) !== 2) {
            top = hull.pop();
        }
        hull.push(top);
        hull.push(points[i]);
    }

    return hull;
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

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}

`.trim();

let currentUrl = "splat_khtest.splat";
// loadSplat 함수 추가
async function loadSplat(url) {
    currentUrl = url;
    await main();
}
async function main() {
    console.log(viewMatrix);

    let carousel = false;
    const params = new URLSearchParams(location.search);
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    // const url = new URL(
    //     // "nike.splat",
    //     // location.href,
    //     params.get("url") || "splat_khtest.splat",
    //     "https://huggingface.co/spatialai/SplatViewer/resolve/main/"
    // );
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
    
    function extractRotationMatrix(viewMatrix) {
        // Ensure the input matrix is a 4x4 matrix
        if (viewMatrix.length !== 16) {
            throw new Error("Input matrix must be a 4x4 matrix.");
        }
    
        // Extract the 3x3 rotation matrix
        return [
            viewMatrix[0], viewMatrix[1], viewMatrix[2],
            viewMatrix[4], viewMatrix[5], viewMatrix[6],
            viewMatrix[8], viewMatrix[9], viewMatrix[10]
        ];
    }
    function updateViewMatrixWithRotation(viewMatrix, newRotationMatrix) { //viewmatrix의 rotationmatrix 교체
        if (viewMatrix.length !== 16 || newRotationMatrix.length !== 9) {
            throw new Error("Invalid matrix dimensions.");
        }
    
        // Create a copy of the original view matrix
        const updatedViewMatrix = [...viewMatrix];
    
        // Update the rotation part of the view matrix
        updatedViewMatrix[0] = newRotationMatrix[0]; // m00
        updatedViewMatrix[1] = newRotationMatrix[1]; // m01
        updatedViewMatrix[2] = newRotationMatrix[2]; // m02
    
        updatedViewMatrix[4] = newRotationMatrix[3]; // m10
        updatedViewMatrix[5] = newRotationMatrix[4]; // m11
        updatedViewMatrix[6] = newRotationMatrix[5]; // m12
    
        updatedViewMatrix[8] = newRotationMatrix[6]; // m20
        updatedViewMatrix[9] = newRotationMatrix[7]; // m21
        updatedViewMatrix[10] = newRotationMatrix[8]; // m22
        // Translation and perspective components remain unchanged
        return updatedViewMatrix;
    }

    let activeKeys = [];
	let currentCameraIndex = 0;

    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    // window.addEventListener(
    //     "wheel",
    //     (e) => {
    //         carousel = false;
    //         e.preventDefault();
    //         const lineHeight = 10;
    //         const scale =
    //             e.deltaMode == 1
    //                 ? lineHeight
    //                 : e.deltaMode == 2
    //                 ? innerHeight
    //                 : 1;
    //         let inv = invert4(viewMatrix);
    //         if (e.shiftKey) {
    //             inv = translate4(
    //                 inv,
    //                 (e.deltaX * scale) / innerWidth,
    //                 (e.deltaY * scale) / innerHeight,
    //                 0,
    //             );
    //         } else if (e.ctrlKey || e.metaKey) {
    //             // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
    //             // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
    //             // let preY = inv[13];
    //             inv = translate4(
    //                 inv,
    //                 0,
    //                 0,
    //                 (-10 * (e.deltaY * scale)) / innerHeight,
    //             );
    //             // inv[13] = preY;
    //         } else {
    //             let d = 4;
    //             inv = translate4(inv, 0, 0, d);
    //             inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
    //             inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
    //             inv = translate4(inv, 0, 0, -d);
    //         }

    //         viewMatrix = invert4(inv);
    //     },
    //     { passive: false },
    // );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });
    
    const sensitivity = 0.1; // 마우스 감도
    let accumulatedRotationX = 0; // 누적된 X축 회전 값
    let accumulatedRotationY = 0; // 누적된 Y축 회전 값

    //default rotationMatrix
    let rotationMatrix = {value: [
        1,0,0,
        0,1,0,
        0,0,1
    ]};
    //초기 틀어짐 방지
    let init_rotationMatrix = {value: [
        1,0,0,
        0,0,-1,
        0,1,0
    ]};
    //space bar 틀어짐 방지
    let viewpoint_rotationMatrix = {value: [
        1,0,0,
        0,1,0,
        0,0,1
    ]};
    //orbit touch 틀어짐 방지
    let orbit_rotationMatrix = {value: [
        1,0,0,
        0,1,0,
        0,0,1
    ]};
    const maxRotationX = 10;
    const minRotationX = -10;
    canvas.addEventListener("mousemove", (e) => {
        // rotation part만 update
        if (!down) return;
        e.preventDefault();
        let inv = invert4(viewMatrix);
        let dx = sensitivity * (e.clientX - startX);
        let dy = sensitivity * (e.clientY - startY);
    
        accumulatedRotationY += dx; // Y축 회전 값 누적 (왼쪽으로 이동하면 증가)
        accumulatedRotationX += dy; // X축 회전 값 누적 (위로 이동하면 증가)
        
        accumulatedRotationX = Math.max(minRotationX, Math.min(accumulatedRotationX, maxRotationX));


        // 회전 행렬 생성
        let rotationX = axisAngleRotationMatrix([1, 0, 0], accumulatedRotationX); // X축 회전
        let rotationY = axisAngleRotationMatrix([0, 1, 0], -accumulatedRotationY); // Y축 회전

        rotationMatrix.value = multiply3x3Matrices(rotationX, rotationY);


        if(spaceStart){ //spacebar 한번 눌리면 나갈 수 없다.

            if(!spaceStartinit){
                spaceStartinit = true;
                accumulatedRotationY = 0; // Y축 회전 값 초기화
                accumulatedRotationX = 0; // X축 회전 값 초기화
                rotationMatrix = createIdentityMatrix3x3(); //기존 rotationMatrix 초기화
            }
            //초기 틀어짐 방지
            rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, viewpoint_rotationMatrix.value);            
            
        }else{ //spacebar눌러서 viewpoint 받기 전
            rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, init_rotationMatrix.value);
        }
        
        //Update
        inv = updateViewMatrixWithRotation(inv, rotationMatrix.value);



        viewMatrix = invert4(inv);
        startX = e.clientX;
        startY = e.clientY;
        // 참고용 원본
        // if (!down) return;
        // e.preventDefault();

        
        // let inv = invert4(viewMatrix);
        // let dx = (50 * (e.clientX - startX)) / innerWidth;
        // let dy = (50 * (e.clientY - startY)) / innerHeight;


        // inv = rotate4(inv, dx, 0, -1, 0);
        // inv = rotate4(inv, -dy, -1, 0, 0);
        // viewMatrix = invert4(inv);
        // //console.log(viewMatrix); 

        // startX = e.clientX;
        // startY = e.clientY;
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        // startX = 0;
        // startY = 0;
    });
    canvas.addEventListener("mouseout", (e) => {
        down = false;
    });
    let altX = 0,
        altY = 0;
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
    let orbitCheck = false;
    let orbitCheckinit = false;
    const dscaleDisplay = document.getElementById('dscale-display');
    dscaleDisplay.textContent = 'Scale: 1';
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);

                const dx = e.touches[0].clientX - startX
                const dy = e.touches[0].clientY - startY

                accumulatedRotationY += dx * 0.1; 
                accumulatedRotationX += dy * 0.1; 
                
                accumulatedRotationX = Math.max(minRotationX, Math.min(accumulatedRotationX, maxRotationX));

                // 회전 행렬 생성
                let rotationX = axisAngleRotationMatrix([1, 0, 0], -accumulatedRotationX); // X축 회전
                let rotationY = axisAngleRotationMatrix([0, 1, 0], accumulatedRotationY); // Y축 회전
        
                rotationMatrix.value = multiply3x3Matrices(rotationX, rotationY);  
        
                if(orbitCheck){
                    console.log("in");
                    if(!orbitCheckinit){
                        orbitCheckinit = true;
                        accumulatedRotationY = 0; // Y축 회전 값 초기화
                        accumulatedRotationX = 0; // X축 회전 값 초기화
                        rotationMatrix = createIdentityMatrix3x3(); //기존 rotationMatrix 초기화
                    }
                    //초기 틀어짐 방지
                    rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, orbit_rotationMatrix.value);  
                    console.log(rotationMatrix.value);


                }else{ //orbit 전
                    rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, init_rotationMatrix.value);
                }

                //Update
                inv = updateViewMatrixWithRotation(inv, rotationMatrix.value);
        
                viewMatrix = invert4(inv);
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                
            } else if (e.touches.length === 2) {

                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                     );
	    	dscaleDisplay.textContent = `Scale: ${dscale.toFixed(2)}`; // 소수점 두 자리까지 표시
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                let tempInv = inv;

                // inv = translate4(inv,  0, 0, d);
                //inv = rotate4(inv, dtheta, 0, 0, 1);

                tempInv = translate4(tempInv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                tempInv = translate4(tempInv, 0, 0, 0.4 * (1 - dscale));
                // inv[13] = preY;

                const tx = tempInv[12];
                const ty = tempInv[14];
                const tz = tempInv[13];
                const tempPositionVector = new Vector3(tx, ty, tz);

                // Check for collision
                if (IsPointInsidePolygon(tempPositionVector)) {
                    // If no collision, update the inverse matrix
                    inv = tempInv;          
                    viewMatrix = invert4(inv);
                } else {
                    console.log('Collision detected, movement blocked.');
                }

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
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
    //없애달라 - 사업부서 의견
    // const joystickRotation = document.getElementById('joystick-rotation');
    // const containerRotation = document.getElementById('joystick-container-rotation');

    // let touchIdRotation = null;
    // let startXRotation = 0;
    // let startYRotation = 0;

    // containerRotation.addEventListener('touchstart', (event) => {
    //     if (touchIdRotation === null) {
    //         const touch = event.changedTouches[0];
    //         touchIdRotation = touch.identifier;
    //         startXRotation = touch.clientX;
    //         startYRotation = touch.clientY;

    //     }
    // });

    // let touchRotSensitivity = 10;
    // containerRotation.addEventListener('touchmove', (event) => {
    //     if (touchIdRotation !== null) {
    //         const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdRotation);
    //         if (touch) {
    //             orbitCheck = true;
    //             let inv = invert4(viewMatrix);
    //             const dx = (touch.clientX - startXRotation) / containerRotation.clientWidth;

    //             //x축 회전은 없앰.
    //             //let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

    //             //orbit 반경 (0:fps)
    //             let d = 0.3;
    //             inv = translate4(inv, 0, 0, d);
    //             inv = rotate4(inv, dx, 0, 1, 0);
    //             inv = translate4(inv, 0, 0, -d);

    //             orbit_rotationMatrix.value = extractRotationMatrix(inv);

    //             viewMatrix = invert4(inv);
    //             moveJoystickMovement2(touch.clientX, touch.clientY);
    //             startXRotation = touch.clientX;
    //             startYRotation = touch.clientY;

    //             // let inv = invert4(viewMatrix);

    //             // const dx = (touch.clientX - startXRotation) / containerRotation.clientWidth;
    //             // const dy = (touch.clientY - startYRotation) / containerRotation.clientHeight;

    //             // accumulatedRotationY += dx * touchRotSensitivity; 
    //             // accumulatedRotationX += dy * touchRotSensitivity; 
                
    //             // accumulatedRotationX = Math.max(minRotationX, Math.min(accumulatedRotationX, maxRotationX));

    //             // // 회전 행렬 생성
    //             // let rotationX = axisAngleRotationMatrix([1, 0, 0], accumulatedRotationX); // X축 회전
    //             // let rotationY = axisAngleRotationMatrix([0, 1, 0], -accumulatedRotationY); // Y축 회전
        
    //             // rotationMatrix.value = multiply3x3Matrices(rotationX, rotationY);  
        
    //             // if(orbitCheck){
    //             //     if(!orbitCheckinit){
    //             //         orbitCheckinit = true;
    //             //         accumulatedRotationY = 0; // Y축 회전 값 초기화
    //             //         accumulatedRotationX = 0; // X축 회전 값 초기화
    //             //         rotationMatrix = createIdentityMatrix3x3(); //기존 rotationMatrix 초기화
    //             //     }
    //             //     //초기 틀어짐 방지
    //             //     rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, orbit_rotationMatrix.value);  
    //             //     console.log(rotationMatrix.value);


    //             // }else{ //spacebar눌러서 viewpoint 받기 전
    //             //     rotationMatrix.value = multiply3x3Matrices(rotationMatrix.value, init_rotationMatrix.value);
    //             // }

    //             // //Update
    //             // inv = updateViewMatrixWithRotation(inv, rotationMatrix.value);
        
    //             // viewMatrix = invert4(inv);
    //             // moveJoystickMovement2(touch.clientX, touch.clientY);
    //             // startXRotation = touch.clientX;
    //             // startYRotation = touch.clientY;
    //         }
    //     }
    // });

    // containerRotation.addEventListener('touchend', (event) => {
    //     const touch = Array.from(event.changedTouches).find(t => t.identifier === touchIdRotation);
    //     if (touch) {
    //         touchIdRotation = null;
    //         resetJoystick()
    //     }
    // });
    // function resetJoystick() {
    //     joystickRotation.style.transform = `translate(-50%, -50%)`;
    // }

    // function moveJoystickMovement2(clientX, clientY) {
    //     const rect = containerRotation.getBoundingClientRect();
    //     const x = clientX - rect.left - rect.width / 2;
    //     const y = clientY - rect.top - rect.height / 2;
    //     const angle = Math.atan2(y, x);
    //     const distance = Math.min(Math.hypot(x, y), rect.width / 2 - joystickRotation.offsetWidth / 2);

    //     const joystickX = distance * Math.cos(angle);
    //     const joystickY = distance * Math.sin(angle);

    //     joystickRotation.style.transform = `translate(${joystickX - 50}%, ${joystickY - 50}%)`;
    // }
    // //조이스틱 구현 끝

    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;

    //spacebar 연속 눌림 방지
    let spacePressed = false;
    //spacebar Start --> Mouse Rotation
    let spaceStart = false;
    //spacebar Start --> Rotation 값 초기화
    let spaceStartinit = false;

    let currentVertexIndex = 0;


    const frame = (now) => {
      
        let inv = invert4(viewMatrix);
        let tempInv = inv;

        if (activeKeys.includes("KeyW")) {
            tempInv = translate4(tempInv, 0, 0, 0.01);
        }
        if (activeKeys.includes("KeyS")) {
            tempInv = translate4(tempInv, 0, 0, -0.01);
        }
        if (activeKeys.includes("KeyA")){
            tempInv = translate4(tempInv, -0.01, 0, 0);
        }
        if (activeKeys.includes("KeyD")){
            tempInv = translate4(tempInv, 0.01, 0, 0);
        }
        if (activeKeys.includes("KeyQ")) tempInv = rotate4(tempInv, 0.1, 0, 0, 1);
        if (activeKeys.includes("KeyE")) tempInv = rotate4(tempInv, -0.01, 0, 0, 1);
        
        const tx = tempInv[12];
        const ty = tempInv[14];
        const tz = tempInv[13];
        const tempPositionVector = new Vector3(tx, ty, tz);
        if (activeKeys.includes("KeyC")) {
            console.log(tempPositionVector); 
            console.log(inv[12],inv[13],inv[14]);
        }
        if (activeKeys.includes("KeyV")){
            console.log(rotationMatrix.value);
        }
        // Check for collision
        if (IsPointInsidePolygon(tempPositionVector)) {
            // If no collision, update the inverse matrix
            inv = tempInv;          
            viewMatrix = invert4(inv);
            console.log();
        } else {
            console.log('Collision detected, movement blocked.');
        }



        if (spacePressed && !noDataCheck) {

            spaceStart = true;
            spaceStartinit = false;
            // 스페이스바를 누를 때마다 현재 인덱스를 증가시키고 배열의 길이로 모듈로 연산을 수행하여 루프를 만듦
            currentVertexIndex = (currentVertexIndex + 1) % polygonVertices.length;
            // 새로운 카메라 위치를 설정
            positionMatrix = translate4(
                positionMatrix,
                polygonVertices[currentVertexIndex].x - positionMatrix[12],
                polygonVertices[currentVertexIndex].y - positionMatrix[14],
                polygonVertices[currentVertexIndex].z - positionMatrix[13]
            );
            positionMatrix[14] = 0;
    
            console.log(currentVertexIndex, 
                positionMatrix[12],
                positionMatrix[14],
                positionMatrix[13]
            );
            // 카메라의 뷰 행렬을 업데이트
            inv = invert4(updateViewMatrix());
            viewpoint_rotationMatrix.value = extractRotationMatrix(inv);
            // 스페이스바 동작을 한 번만 수행하도록 설정
            spacePressed = false;
        }

        // 키 누름 이벤트 핸들러
        window.addEventListener('keydown', (event) => {
            if (!activeKeys.includes(event.code)) {
                activeKeys.push(event.code);
            }
            if (event.code === "Space" && !spacePressed) {
                spacePressed = true;
            }
        });

        // 키 놓음 이벤트 핸들러
        window.addEventListener('keyup', (event) => {
            const index = activeKeys.indexOf(event.code);
            if (index > -1) {
                activeKeys.splice(index, 1);
            }
        });
        inv[14] = 0.01; // 14번째 요소 (0부터 시작) = Y축의 위치 고정

        viewMatrix = invert4(inv);
       
        let inv2 = invert4(viewMatrix);

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

main().catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
