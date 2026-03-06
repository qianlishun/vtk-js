import macro from 'vtk.js/Sources/macros';
import vtkPolyData from 'vtk.js/Sources/Common/DataModel/PolyData';

const SMOOTH_WGSL = `
struct Params {
  coeff: f32,
  numPoints: f32, // 统一用 f32 传入，在内部转 u32，避免 JS/GPU 字节序解析错误
  stride: f32,
  padding: f32,
};

@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var<storage, read> neighbors : array<i32>;
@group(0) @binding(2) var<storage, read> posIn : array<f32>;
@group(0) @binding(3) var<storage, read_write> posOut : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  let numPts = u32(p.numPoints);
  if (i >= numPts) { return; }

  let base = i * 3u;
  let stride = u32(p.stride);
  let nStart = i * stride;
  let count = u32(neighbors[nStart]);
  
  let curPos = vec3<f32>(posIn[base], posIn[base + 1u], posIn[base + 2u]);

  // 必须初始化 posOut，防止 count 为 0 时输出随机值
  if (count == 0u) {
    posOut[base] = curPos.x;
    posOut[base + 1u] = curPos.y;
    posOut[base + 2u] = curPos.z;
    return;
  }

  var avg = vec3<f32>(0.0);
  for (var j = 1u; j <= count; j++) {
    if (j >= stride) { break; } // 安全阈值
    let nIdx = u32(neighbors[nStart + j]);
    if (nIdx < numPts) {
      let nBase = nIdx * 3u;
      avg += vec3<f32>(posIn[nBase], posIn[nBase + 1u], posIn[nBase + 2u]);
    }
  }
  avg /= f32(count);

  let newPos = curPos + p.coeff * (avg - curPos);

  // 写入检查：防止产生 NaN 导致图像消失或碎裂
  if (all(finite(newPos))) {
    posOut[base] = newPos.x;
    posOut[base + 1u] = newPos.y;
    posOut[base + 2u] = newPos.z;
  } else {
    posOut[base] = curPos.x;
    posOut[base + 1u] = curPos.y;
    posOut[base + 2u] = curPos.z;
  }
}

// 辅助函数
fn finite(v: vec3<f32>) -> vec3<bool> {
  return vec3<bool>(
    !(v.x != v.x), // NaN 检查
    !(v.y != v.y),
    !(v.z != v.z)
  );
}
`;

function vtkSimpleGPUSmoothFilter(publicAPI, model) {
  model.classHierarchy.push('vtkSimpleGPUSmoothFilter');

  // --- CPU 部分：构建邻接表 ---
  publicAPI.buildAdjacency = (polyData) => {
    const polys = polyData.getPolys().getData();
    const numPoints = polyData.getNumberOfPoints();
    const adj = Array.from({ length: numPoints }, () => new Set());

    let offset = 0;
    while (offset < polys.length) {
      const n = polys[offset];
      for (let i = 1; i <= n; i++) {
        const v1 = polys[offset + i];
        const v2 = polys[offset + (i % n) + 1];
        adj[v1].add(v2);
        adj[v2].add(v1);
      }
      offset += n + 1;
    }

    let maxN = 0;
    adj.forEach((s) => {
      maxN = Math.max(maxN, s.size);
    });

    // stride = 1 (存储邻居数量) + maxN (存储邻居索引)
    const stride = maxN + 1;
    model.stride = stride; // 存入 model 供循环使用

    const buffer = new Int32Array(numPoints * stride);
    for (let i = 0; i < numPoints; i++) {
      const neighbors = Array.from(adj[i]);
      buffer[i * stride] = neighbors.length;
      for (let j = 0; j < neighbors.length; j++) {
        buffer[i * stride + j + 1] = neighbors[j];
      }
    }
    return buffer;
  };

  // --- GPU 部分：执行计算 ---
  publicAPI.requestData = (inData, outData) => {
    const input = inData[0];
    if (model.numberOfIterations === 0) {
      outData[0] = input;
      return;
    }
    model.device = model.webGPURenderWindow.getDevice();
    if (!model.device) {
      return;
    }
    if (!input || model.isBusy) {
      return;
    }
    const runCompute = async () => {
      try {
        console.time('SimpleGPUSmoothFilter');

        model.isBusy = true;
        // Windowed Sinc 系数计算 (基于 passBand)
        // eslint-disable-next-line camelcase
        const lambda = 0.33;
        const theta = model.passBand || 0.1;
        // 核心：防止分母接近 0
        // eslint-disable-next-line prettier/prettier
        const denominator = ( 1.0 / theta ) - ( 1.0 / lambda );
        const mu = Math.abs(denominator) < 1e-6 ? -0.34 : -1.0 / denominator;

        const points = input.getPoints().getData();
        const numPoints = input.getNumberOfPoints();

        console.time('buildAdjacency');
        const adjBuffer = publicAPI.buildAdjacency(input);
        console.timeEnd('buildAdjacency');
        const stride = model.stride; // 现在可以安全访问了

        // 初始化 GPU 管线
        if (!model.pipeline) {
          model.shaderModule = model.device.getHandle().createShaderModule({
            code: SMOOTH_WGSL,
          });
          model.pipeline = model.device.getHandle().createComputePipeline({
            layout: 'auto',
            compute: { module: model.shaderModule, entryPoint: 'main' },
          });
        }

        // 创建存储 Buffers
        const createStorageBuffer = (data) => {
          const b = model.device.getHandle().createBuffer({
            size: data.byteLength,
            usage:
              // eslint-disable-next-line no-bitwise, no-undef
              GPUBufferUsage.STORAGE |
              // eslint-disable-next-line no-undef
              GPUBufferUsage.COPY_SRC |
              // eslint-disable-next-line no-undef
              GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
          });
          new data.constructor(b.getMappedRange()).set(data);
          b.unmap();
          return b;
        };

        const bNeighbors = createStorageBuffer(adjBuffer);
        const bPing = createStorageBuffer(points);
        const bPong = createStorageBuffer(points);

        const numGroups = Math.ceil(numPoints / 64);

        // 创建两个独立的 Uniform Buffer
        const bParamsLambda = model.device.getHandle().createBuffer({
          size: 16, // [coeff, numPoints, stride, padding]
          // eslint-disable-next-line no-bitwise, no-undef
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bParamsMu = model.device.getHandle().createBuffer({
          size: 16,
          // eslint-disable-next-line no-bitwise, no-undef
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        model.device
          .getHandle()
          .queue.writeBuffer(
            bParamsLambda,
            0,
            new Float32Array([lambda, numPoints, stride, 0])
          );
        model.device
          .getHandle()
          .queue.writeBuffer(
            bParamsMu,
            0,
            new Float32Array([mu, numPoints, stride, 0])
          );

        const bindGroupPing = model.device.getHandle().createBindGroup({
          layout: model.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: bParamsLambda } },
            { binding: 1, resource: { buffer: bNeighbors } },
            { binding: 2, resource: { buffer: bPing } },
            { binding: 3, resource: { buffer: bPong } },
          ],
        });

        // BindGroup 2: 使用 Mu 参数，将 Pong 计算到 Ping
        const bindGroupPong = model.device.getHandle().createBindGroup({
          layout: model.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: bParamsMu } }, // 使用 Mu
            { binding: 1, resource: { buffer: bNeighbors } },
            { binding: 2, resource: { buffer: bPong } },
            { binding: 3, resource: { buffer: bPing } },
          ],
        });

        for (let i = 0; i < model.numberOfIterations; i++) {
          // 第一步：Lambda Pass (bPing -> bPong)
          const encoder = model.device.getHandle().createCommandEncoder();

          const pass1 = encoder.beginComputePass();
          pass1.setPipeline(model.pipeline);
          pass1.setBindGroup(0, bindGroupPing); // 绑定 bPing 为输入，bPong 为输出
          pass1.dispatchWorkgroups(numGroups);
          pass1.end();

          // 第二步：Mu Pass (bPong -> bPing)
          const pass2 = encoder.beginComputePass();
          pass2.setPipeline(model.pipeline);
          pass2.setBindGroup(0, bindGroupPong); // 绑定 bPong 为输入，bPing 为输出
          pass2.dispatchWorkgroups(numGroups);
          pass2.end();

          model.device.getHandle().queue.submit([encoder.finish()]);
        }

        await model.device.getHandle().queue.onSubmittedWorkDone();

        // 读取结果并更新 PolyData
        const readBuffer = model.device.getHandle().createBuffer({
          size: points.byteLength,
          // eslint-disable-next-line no-bitwise, no-undef
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const readEncoder = model.device.getHandle().createCommandEncoder();
        readEncoder.copyBufferToBuffer(
          bPing,
          0,
          readBuffer,
          0,
          points.byteLength
        );
        model.device.getHandle().queue.submit([readEncoder.finish()]);

        // eslint-disable-next-line no-undef
        await readBuffer.mapAsync(GPUMapMode.READ);
        console.time('Map slice');
        const finalPoints = new Float32Array(
          readBuffer.getMappedRange().slice(0, points.byteLength)
        );
        readBuffer.unmap();
        console.timeEnd('Map slice');

        const output = vtkPolyData.newInstance();
        output.shallowCopy(input);
        output.getPoints().setData(finalPoints);

        outData[0] = output;

        console.timeEnd('SimpleGPUSmoothFilter');
      } finally {
        model.isBusy = false;
      }
    };
    runCompute();
  };
}

const DEFAULT_VALUES = {
  numberOfIterations: 20,
  passBand: 0.1,
  device: null,
  webGPURenderWindow: null,
};

export function extend(publicAPI, model, initialValues = {}) {
  Object.assign(model, DEFAULT_VALUES, initialValues);
  macro.obj(publicAPI, model);
  macro.algo(publicAPI, model, 1, 1);
  macro.setGet(publicAPI, model, [
    'numberOfIterations',
    'passBand',
    'webGPURenderWindow',
    'device',
  ]);
  vtkSimpleGPUSmoothFilter(publicAPI, model);
}

export const newInstance = macro.newInstance(
  extend,
  'vtkSimpleGPUSmoothFilter'
);

export default { newInstance, extend };
