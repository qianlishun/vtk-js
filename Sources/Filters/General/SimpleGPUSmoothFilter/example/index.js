import '@kitware/vtk.js/favicon';

// Load the rendering pieces we want to use (for both WebGL and WebGPU)
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
// import vtkWindowedSincPolyDataFilter from '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter';
import vtkSimpleGPUSmoothFilter from '@kitware/vtk.js/Filters/General/SimpleGPUSmoothFilter';
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPolyDataNormals from '@kitware/vtk.js/Filters/Core/PolyDataNormals';

import GUI from 'lil-gui';

// Force DataAccessHelper to have access to various data source
import '@kitware/vtk.js/IO/Core/DataAccessHelper/HttpDataAccessHelper';

function createVTKImageData(typedArray, dims, dataType, spacing) {
  const data = vtkImageData.newInstance();
  const [width, height, depth] = dims;
  data.setDimensions(width, height, depth);
  data.setSpacing(...spacing);

  const scalars = vtkDataArray.newInstance({
    values: typedArray,
    numberOfComponents: 1,
    dataType,
  });
  data.getPointData().setScalars(scalars);

  return data;
}

// Create three spheres for testing
function generateTestVolumeData(size) {
  const WIDTH = size[0];
  const HEIGHT = size[1];
  const DEPTH = size[2];
  const TOTAL_COUNT = WIDTH * HEIGHT * DEPTH;
  const data = new Uint8Array(TOTAL_COUNT);

  const centerX1 = WIDTH / 2;
  const centerY1 = HEIGHT / 2;
  const centerZ1 = DEPTH / 2;
  const radius = Math.min(WIDTH, HEIGHT, DEPTH) / 3;

  for (let z = 0; z < DEPTH; z++) {
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        const index = z * WIDTH * HEIGHT + y * WIDTH + x;
        const distance = Math.sqrt(
          (x - centerX1) ** 2 + (y - centerY1) ** 2 + (z - centerZ1) ** 2
        );

        if (distance < radius) {
          data[index] = 150 + Math.random() * 10;
        } else {
          data[index] = Math.random() * 10;
        }
      }
    }
  }

  return data;
}

const dataSize = [64, 64, 64];
// eslint-disable-next-line no-use-before-define
const testData = generateTestVolumeData(dataSize);
const originalImageData = createVTKImageData(
  testData,
  dataSize,
  'Uint8Array',
  [1, 1, 1]
);

// ----------------------------------------------------------------------------
// Standard rendering code setup
// ----------------------------------------------------------------------------
const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
  defaultViewAPI: 'WebGPU',
  background: [0.1, 0.1, 0.2],
});
const apiRW = fullScreenRenderer.getApiSpecificRenderWindow();
console.log('SpecificRenderWindow', apiRW.getClassName());

const renderer = fullScreenRenderer.getRenderer();
const renderWindow = fullScreenRenderer.getRenderWindow();

// ----------------------------------------------------------------------------
// Example code
// ----------------------------------------------------------------------------

const actor = vtkActor.newInstance();
renderer.addActor(actor);

const mapper = vtkMapper.newInstance({ interpolateScalarBeforeMapping: true });
actor.setMapper(mapper);

const property = actor.getProperty();
property.setAmbient(0.2);
property.setDiffuse(0.8);
property.setSpecular(0.1);
property.setSpecularPower(30);
property.setColor(0.7, 0.5, 0.3); // 金色

const camera = renderer.getActiveCamera();
camera.setPosition(1, 1, -1);
camera.setViewUp(0, 1, 0);
camera.setFocalPoint(0, 0, 0);

// 1. 等值面提取（Marching Cubes）
const marchingCube = vtkImageMarchingCubes.newInstance({
  contourValue: 150,
  computeNormals: true,
  mergePoints: true,
});
marchingCube.setInputData(originalImageData);

const smoothFilter = vtkSimpleGPUSmoothFilter.newInstance({
  numberOfIterations: 0,
  passBand: 0.1,
});
smoothFilter.setWebGPURenderWindow(apiRW);
smoothFilter.setInputConnection(marchingCube.getOutputPort());

const normalFilter = vtkPolyDataNormals.newInstance();
normalFilter.setInputConnection(smoothFilter.getOutputPort());
normalFilter.setComputePointNormals(true);
normalFilter.setComputeCellNormals(true);

mapper.setInputConnection(normalFilter.getOutputPort());

// ----------------------------------------------------------------------------
// UI control handling
// ----------------------------------------------------------------------------

const gui = new GUI();
const params = {
  numberOfIterations: 0,
  passBand: 0.1,
};
gui
  .add(params, 'numberOfIterations', 0, 100, 1)
  .name('Iterations')
  .onChange((v) => {
    smoothFilter.set({ numberOfIterations: Number(v) });
    renderWindow.render();
  });
gui
  .add(params, 'passBand', 0.1, 1, 0.05)
  .name('Pass band')
  .onChange((v) => {
    const value = 10.0 ** (-4.0 * Number(v));
    smoothFilter.set({ passBand: value });
    renderWindow.render();
  });

// -----------------------------------------------------------

renderer.resetCamera();
renderWindow.render();

// -----------------------------------------------------------
// Make some variables global so that you can inspect and
// modify objects in your browser's developer console:
// -----------------------------------------------------------

global.source = originalImageData;
global.filter = smoothFilter;
global.mapper = mapper;
global.actor = actor;
