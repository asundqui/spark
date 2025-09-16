import * as THREE from "three";
import { Readback, type SplatGenerator, dyno } from ".";
import { NewSplatAccumulator } from "./NewSplatAccumulator";
import {
  DEFAULT_SPLAT_ENCODING,
  DynoPackedSplats,
  PackedSplats,
  type SplatEncoding,
} from "./PackedSplats";
import type { GeneratorMapping } from "./SplatAccumulator";
import { SplatGeometry } from "./SplatGeometry";
import { SplatMesh } from "./SplatMesh";
import {
  EXT_LN_SCALE_MAX,
  EXT_LN_SCALE_MIN,
  LN_SCALE_MAX,
  LN_SCALE_MIN,
} from "./defines";
import { getShaders } from "./shaders";
import { withWorker } from "./splatWorker";
import { FreeList } from "./utils";

// // Scene.onBeforeRender monkey-patch to
// // inject a NewSparkRenderer into a scene with SplatMeshes if there isn't
// // one already. Restore original Scene.onBeforeRenderer and Scene.add when done.
// let hasSplatMesh = false;
// let hasSparkRenderer = false;

// let sparkRendererInstance: NewSparkRenderer;

// function containsSplatMesh(object3D: THREE.Object3D) {
//   let hasSplatMesh = false;
//   if (object3D instanceof SplatMesh) {
//     return true;
//   }
//   object3D.traverse((child: THREE.Object3D) => {
//     hasSplatMesh = hasSplatMesh || child instanceof SplatMesh;
//   });
//   return hasSplatMesh;
// }

// const sceneAdd = THREE.Scene.prototype.add;
// THREE.Scene.prototype.add = function (object: THREE.Object3D) {
//   hasSplatMesh = hasSplatMesh || containsSplatMesh(object);
//   hasSparkRenderer = hasSparkRenderer || object instanceof NewSparkRenderer;
//   sceneAdd.call(this, object);
//   return this;
// };

// const sceneOnBeforeRender = THREE.Scene.prototype.onBeforeRender;
// THREE.Scene.prototype.onBeforeRender = function (
//   renderer: THREE.WebGLRenderer,
// ) {
//   if (!hasSplatMesh) {
//     return;
//   }
//   if (!hasSparkRenderer) {
//     const spark = sparkRendererInstance || new NewSparkRenderer({ renderer });
//     this.add(spark);
//   }
//   THREE.Scene.prototype.onBeforeRender = sceneOnBeforeRender;
//   THREE.Scene.prototype.add = sceneAdd;
// };

export interface NewSparkRendererOptions {
  /**
   * Pass in your THREE.WebGLRenderer instance so Spark can perform work
   * outside the usual render loop. Should be created with antialias: false
   * (default setting) as WebGL anti-aliasing doesn't improve Gaussian Splatting
   * rendering and significantly reduces performance.
   */
  renderer: THREE.WebGLRenderer;
  /**
   * Whether to use premultiplied alpha when accumulating splat RGB
   * @default true
   */
  premultipliedAlpha?: boolean;
  /**
   * Controls whether to check and automatically update Gsplat collection after
   * each frame render.
   * @default true
   */
  autoUpdate?: boolean;
  /**
   * Controls whether to update the Gsplats before or after rendering. For WebXR
   * this must be false in order to complete rendering as soon as possible.
   * @default true
   */
  preUpdate?: boolean;
  /**
   * Maximum standard deviations from the center to render Gaussians. Values
   * Math.sqrt(5)..Math.sqrt(8) produce good results and can be tweaked for
   * performance.
   * @default Math.sqrt(8)
   */
  maxStdDev?: number;
  /*
   **
   * Minimum pixel radius for splat rendering.
   * @default 0.0
   */
  minPixelRadius?: number;
  /**
   * Maximum pixel radius for splat rendering.
   * @default 512.0
   */
  maxPixelRadius?: number;
  /**
   * Minimum alpha value for splat rendering.
   * @default 0.5 * (1.0 / 255.0)
   */
  minAlpha?: number;
  /**
   * Enable 2D Gaussian splatting rendering ability. When this mode is enabled,
   * any scale x/y/z component that is exactly 0 (minimum quantized value) results
   * in the other two non-0 axis being interpreted as an oriented 2D Gaussian Splat,
   * rather instead of the usual projected 3DGS Z-slice. When reading PLY files,
   * scale values less than e^-30 will be interpreted as 0.
   * @default false
   */
  enable2DGS?: boolean;
  /**
   * Scalar value to add to 2D splat covariance diagonal, effectively blurring +
   * enlarging splats. In scenes trained without the Gsplat anti-aliasing tweak
   * this value was typically 0.3, but with anti-aliasing it is 0.0
   * @default 0.0
   */
  preBlurAmount?: number;
  /**
   * Scalar value to add to 2D splat covarianve diagonal, with opacity adjustment
   * to correctly account for "blurring" when anti-aliasing. Typically 0.3
   * (equivalent to approx 0.5 pixel radius) in scenes trained with anti-aliasing.
   */
  blurAmount?: number;
  /**
   * Depth-of-field distance to focal plane
   */
  focalDistance?: number;
  /**
   * Full-width angle of aperture opening (in radians), 0.0 to disable
   * @default 0.0
   */
  apertureAngle?: number;
  /**
   * Modulate Gaussian kernel falloff. 0 means "no falloff, flat shading",
   * while 1 is the normal Gaussian kernel.
   * @default 1.0
   */
  falloff?: number;
  /**
   * X/Y clipping boundary factor for Gsplat centers against view frustum.
   * 1.0 clips any centers that are exactly out of bounds, while 1.4 clips
   * centers that are 40% beyond the bounds.
   * @default 1.4
   */
  clipXY?: number;
  /**
   * Parameter to adjust projected splat scale calculation to match other renderers,
   * similar to the same parameter in the MKellogg 3DGS renderer. Higher values will
   * tend to sharpen the splats. A value 2.0 can be used to match the behavior of
   * the PlayCanvas renderer.
   * @default 1.0
   */
  focalAdjustment?: number;
  /**
   * Whether to encode Gsplat with linear RGB (for environment mapping)
   * @default false
   */
  encodeLinear?: boolean;
  /**
   * Override the default splat encoding ranges for the PackedSplats.
   * (default: undefined)
   */
  splatEncoding?: SplatEncoding;
  /**
   * Whether to sort splats radially (geometric distance) from the viewpoint (true)
   * or by Z-depth (false). Most scenes are trained with the Z-depth sort metric
   * and will render more accurately at certain viewpoints. However, radial sorting
   * is more stable under viewpoint rotations.
   * @default true
   */
  sortRadial?: boolean;
  /**
   * Constant added to Z-depth to bias values into the positive range for
   * sortRadial: false, but also used for culling Gsplats "well behind"
   * the viewpoint origin
   * @default 1.0
   */
  depthBias?: number;
  /**
   * Set this to true if rendering a 360 to disable "behind the viewpoint"
   * culling during sorting. This is set automatically when rendering 360 envMaps
   * using the SparkRenderer.renderEnvMap() utility function.
   * @default false
   */
  sort360?: boolean;
}

export class NewSparkRenderer extends THREE.Mesh {
  renderer: THREE.WebGLRenderer;
  premultipliedAlpha: boolean;
  material: THREE.ShaderMaterial;
  uniforms: ReturnType<typeof NewSparkRenderer.makeUniforms>;

  autoUpdate: boolean;
  preUpdate: boolean;

  maxStdDev: number;
  minPixelRadius: number;
  maxPixelRadius: number;
  minAlpha: number;
  enable2DGS: boolean;
  preBlurAmount: number;
  blurAmount: number;
  focalDistance: number;
  apertureAngle: number;
  falloff: number;
  clipXY: number;
  focalAdjustment: number;
  encodeLinear: boolean;
  splatEncoding: SplatEncoding;

  sortRadial?: boolean;
  depthBias?: number;
  sort360?: boolean;

  active: NewSplatAccumulator;
  pending: NewSplatAccumulator;
  sortMapping: Map<SplatGenerator, GeneratorMapping>;
  sortDirty: boolean;
  sorting: boolean;
  readback: Uint32Array;
  orderingFreelist: FreeList<Uint32Array, number>;
  // pendingGeometry: SplatGeometry;

  private lastFrame = -1;

  constructor(options: NewSparkRendererOptions) {
    const uniforms = NewSparkRenderer.makeUniforms();
    const shaders = getShaders();
    const premultipliedAlpha = options.premultipliedAlpha ?? true;
    const geometry = new SplatGeometry(new Uint32Array(1), 0);
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: shaders.splatVertex,
      fragmentShader: shaders.splatFragment,
      uniforms,
      premultipliedAlpha,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    super(geometry, material);
    this.material = material;
    this.uniforms = uniforms;
    // Disable frustum culling because we want to always draw them all
    // and cull Gsplats individually in the shader
    this.frustumCulled = false;

    // sparkRendererInstance = this;
    this.renderer = options.renderer;
    this.premultipliedAlpha = premultipliedAlpha;
    this.autoUpdate = options.autoUpdate ?? true;
    this.preUpdate = options.preUpdate ?? true;

    this.maxStdDev = options.maxStdDev ?? Math.sqrt(8.0);
    this.minPixelRadius = options.minPixelRadius ?? 1.6;
    this.maxPixelRadius = options.maxPixelRadius ?? 512.0;
    this.minAlpha = options.minAlpha ?? 0.5 * (1.0 / 255.0);
    this.enable2DGS = options.enable2DGS ?? false;
    this.preBlurAmount = options.preBlurAmount ?? 0.0;
    this.blurAmount = options.blurAmount ?? 0.3;
    this.focalDistance = options.focalDistance ?? 0.0;
    this.apertureAngle = options.apertureAngle ?? 0.0;
    this.falloff = options.falloff ?? 1.0;
    this.clipXY = options.clipXY ?? 1.4;
    this.focalAdjustment = options.focalAdjustment ?? 1.0;
    this.encodeLinear = options.encodeLinear ?? false;
    this.splatEncoding = options.splatEncoding ?? { ...DEFAULT_SPLAT_ENCODING };

    this.sortRadial = options.sortRadial;
    this.depthBias = options.depthBias;
    this.sort360 = options.sort360;

    this.active = new NewSplatAccumulator({
      splatEncoding: this.splatEncoding,
    });
    this.pending = new NewSplatAccumulator({
      splatEncoding: this.splatEncoding,
    });
    this.sortMapping = new Map();
    this.sortDirty = false;
    this.sorting = false;
    this.readback = new Uint32Array(0);
    this.orderingFreelist = new FreeList<Uint32Array, number>({
      allocate: (maxSplats) => new Uint32Array(maxSplats),
      valid: (ordering, maxSplats) => ordering.length === maxSplats,
    });
    // this.pendingGeometry = new SplatGeometry(new Uint32Array(1), 0);
  }

  static makeUniforms() {
    const uniforms = {
      // Size of render viewport in pixels
      renderSize: { value: new THREE.Vector2() },
      // Near and far plane distances
      near: { value: 0.1 },
      far: { value: 1000.0 },
      // Total number of Gsplats in packedSplats to render
      numSplats: { value: 0 },
      // SplatAccumulator to view transformation quaternion
      renderToViewQuat: { value: new THREE.Quaternion() },
      // SplatAccumulator to view transformation translation
      renderToViewPos: { value: new THREE.Vector3() },
      // Maximum distance (in stddevs) from Gsplat center to render
      maxStdDev: { value: 1.0 },
      // Minimum pixel radius for splat rendering
      minPixelRadius: { value: 1.6 },
      // Maximum pixel radius for splat rendering
      maxPixelRadius: { value: 512.0 },
      // Minimum alpha value for splat rendering
      minAlpha: { value: 0.5 * (1.0 / 255.0) },
      // Enable interpreting 0-thickness Gsplats as 2DGS
      enable2DGS: { value: false },
      // Add to projected 2D splat covariance diagonal (thickens and brightens)
      preBlurAmount: { value: 0.0 },
      // Add to 2D splat covariance diagonal and adjust opacity (anti-aliasing)
      blurAmount: { value: 0.3 },
      // Depth-of-field distance to focal plane
      focalDistance: { value: 0.0 },
      // Full-width angle of aperture opening (in radians)
      apertureAngle: { value: 0.0 },
      // Modulate Gaussian kernal falloff. 0 means "no falloff, flat shading",
      // 1 is normal e^-x^2 falloff.
      falloff: { value: 1.0 },
      // Clip Gsplats that are clipXY times beyond the +-1 frustum bounds
      clipXY: { value: 1.4 },
      // Debug renderSize scale factor
      focalAdjustment: { value: 1.0 },
      // Whether to encode Gsplat with linear RGB (for environment mapping)
      encodeLinear: { value: false },
      // Gsplat collection to render
      packedSplats: { type: "t", value: PackedSplats.getEmpty() },
      // Gsplat collection to render
      packedSplats2: { type: "t", value: PackedSplats.getEmpty() },
      // Whether to use extended splat encoding
      extended: { value: false },
      // Splat encoding ranges
      rgbMinMaxLnScaleMinMax: { value: new THREE.Vector4() },
      // Time in seconds for time-based effects
      time: { value: 0 },
      // Delta time in seconds since last frame
      deltaTime: { value: 0 },
      // Debug flag that alternates each frame
      debugFlag: { value: false },
      numIndexMapping: { value: 0 },
      indexMapping: { value: new Uint32Array(64 * 4) },
    };
    return uniforms;
  }

  onBeforeRender(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ) {
    const frame = renderer.info.render.frame;
    const isNewFrame = frame !== this.lastFrame;
    this.lastFrame = frame;

    if (this.autoUpdate && this.preUpdate && isNewFrame) {
      this.update({ renderer, scene, camera });
    }

    const renderSize = renderer.getDrawingBufferSize(
      this.uniforms.renderSize.value,
    );

    const typedCamera = camera as
      | THREE.PerspectiveCamera
      | THREE.OrthographicCamera;

    this.uniforms.near.value = typedCamera.near;
    this.uniforms.far.value = typedCamera.far;

    const geometry = this.geometry as SplatGeometry;
    this.uniforms.numSplats.value = geometry.instanceCount;

    const accumToWorld = this.active.originToWorld.clone();
    const worldToCamera = camera.matrixWorld.clone().invert();
    const originToCamera = accumToWorld.premultiply(worldToCamera);
    originToCamera.decompose(
      this.uniforms.renderToViewPos.value,
      this.uniforms.renderToViewQuat.value,
      new THREE.Vector3(),
    );

    this.uniforms.maxStdDev.value = this.maxStdDev;
    this.uniforms.minPixelRadius.value = this.minPixelRadius;
    this.uniforms.maxPixelRadius.value = this.maxPixelRadius;
    this.uniforms.minAlpha.value = this.minAlpha;
    this.uniforms.enable2DGS.value = this.enable2DGS;
    this.uniforms.preBlurAmount.value = this.preBlurAmount;
    this.uniforms.blurAmount.value = this.blurAmount;
    this.uniforms.focalDistance.value = this.focalDistance;
    this.uniforms.apertureAngle.value = this.apertureAngle;
    this.uniforms.falloff.value = this.falloff;
    this.uniforms.clipXY.value = this.clipXY;
    this.uniforms.focalAdjustment.value = this.focalAdjustment;
    this.uniforms.encodeLinear.value = this.encodeLinear;

    const texture = this.active.splats.getTexture();
    const extended = Array.isArray(texture);
    this.uniforms.packedSplats.value = extended ? texture[0] : texture;
    this.uniforms.packedSplats2.value = extended
      ? texture[1]
      : PackedSplats.getEmpty();
    this.uniforms.extended.value = extended;
    this.uniforms.rgbMinMaxLnScaleMinMax.value.set(
      this.active.splats.splatEncoding?.rgbMin ?? 0.0,
      this.active.splats.splatEncoding?.rgbMax ?? 1.0,
      this.active.splats.splatEncoding?.lnScaleMin ??
        (extended ? EXT_LN_SCALE_MIN : LN_SCALE_MIN),
      this.active.splats.splatEncoding?.lnScaleMax ??
        (extended ? EXT_LN_SCALE_MAX : LN_SCALE_MAX),
    );
    this.uniforms.time.value = this.active.time;

    // Alternating debug flag that can aid in visual debugging
    this.uniforms.debugFlag.value = (performance.now() / 1000.0) % 2.0 < 1.0;

    const numIndexMapping = this.sortMapping.size;
    this.uniforms.numIndexMapping.value = numIndexMapping;
    let i = 0;
    for (const mapping of this.sortMapping.values()) {
      const active = this.active.mapping.get(mapping.node);
      this.uniforms.indexMapping.value[4 * i + 0] = mapping.base;
      this.uniforms.indexMapping.value[4 * i + 1] = mapping.count;
      this.uniforms.indexMapping.value[4 * i + 2] = active ? active.base : 0;
      this.uniforms.indexMapping.value[4 * i + 3] = active ? active.count : 0;
      ++i;
    }
  }

  update({
    renderer,
    scene,
    camera,
  }: {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.Camera;
  }) {
    this.updateMatrixWorld();
    const originToWorld = this.matrixWorld;
    const renderSize = this.uniforms.renderSize.value;
    this.pending.generateSplats({
      renderer,
      scene,
      sortMapping: this.sortMapping,
      lastSplats: this.active,
      originToWorld,
      camera,
      renderSize,
    });
    this.sortDirty = true;

    const oldActive = this.active;
    this.active = this.pending;
    this.pending = oldActive;

    // Don't await this
    this.driveSort();
  }

  async driveSort() {
    if (this.sorting || !this.sortDirty) {
      return;
    }
    try {
      this.sorting = true;

      const { numSplats, maxSplats } = this.active.splats;
      let activeSplats = 0;
      let ordering = this.orderingFreelist.alloc(maxSplats);
      this.readback = reader.ensureBuffer(maxSplats, this.readback);
      // console.log(`maxSplats = ${maxSplats}`);

      const worldToOrigin = this.active.originToWorld.clone().invert();
      const viewToOrigin = this.active.viewToWorld
        .clone()
        .premultiply(worldToOrigin);

      dynoSort360.value = this.sort360 ?? true;
      dynoSortRadial.value = dynoSort360.value
        ? true
        : (this.sortRadial ?? true);
      dynoOrigin.value.set(0, 0, 0).applyMatrix4(viewToOrigin);
      dynoDirection.value
        .set(0, 0, -1)
        .applyMatrix4(viewToOrigin)
        .sub(dynoOrigin.value)
        .normalize();
      dynoDepthBias.value = this.depthBias ?? 1.0;
      dynoSplats.packedSplats = this.active.splats;
      // console.log(`dynoSplats.packedSplats = ${dynoSplats.packedSplats.maxSplats}, ${dynoSplats.packedSplats.numSplats}`);

      const sortMapping = this.active.mapping;
      this.sortDirty = false;

      await reader.renderReadback({
        renderer: this.renderer,
        reader: sortReader,
        count: numSplats,
        readback: this.readback,
      });
      // console.log(`readback = ${this.readback.length}`);
      // console.log(`readback = ${this.readback.slice(0, 10)}`);

      const result = (await withWorker(async (worker) => {
        return worker.call("sort32Splats", {
          maxSplats,
          numSplats,
          readback: this.readback,
          ordering,
        });
      })) as {
        readback: Uint16Array | Uint32Array;
        ordering: Uint32Array;
        activeSplats: number;
      };

      // // Add delay to sort
      // await new Promise(resolve => setTimeout(resolve, 500));

      this.readback = result.readback as Uint32Array;
      ordering = result.ordering;
      activeSplats = result.activeSplats;
      // console.log(`activeSplats = ${activeSplats}, ordering = ${ordering.slice(0, 10)}`);
      console.log(`activeSplats = ${activeSplats}`);

      // const oldOrdering = this.pendingGeometry.ordering;
      // if (oldOrdering.length === ordering.length) {
      //   this.pendingGeometry.update(ordering, activeSplats);
      // } else {
      //   this.pendingGeometry.dispose();
      //   this.pendingGeometry = new SplatGeometry(ordering, activeSplats);
      // }
      // this.orderingFreelist.free(oldOrdering);

      // const oldGeometry = this.geometry as SplatGeometry;
      // this.geometry = this.pendingGeometry;
      // this.pendingGeometry = oldGeometry;

      const oldOrdering = (this.geometry as SplatGeometry).ordering;
      this.geometry.dispose();
      this.orderingFreelist.free(oldOrdering);

      this.geometry = new SplatGeometry(ordering, activeSplats);

      this.sortMapping = sortMapping;
      this.material.needsUpdate = true;
    } finally {
      this.sorting = false;
    }

    // Don't await this
    this.driveSort();
  }
}

const dynoSortRadial = new dyno.DynoBool({ value: true });
const dynoOrigin = new dyno.DynoVec3({ value: new THREE.Vector3() });
const dynoDirection = new dyno.DynoVec3({ value: new THREE.Vector3() });
const dynoDepthBias = new dyno.DynoFloat({ value: 1.0 });
const dynoSort360 = new dyno.DynoBool({ value: false });
const dynoSplats = new DynoPackedSplats();

const reader = new Readback();
const sortReader = dyno.dynoBlock(
  { index: "int" },
  { rgba8: "vec4" },
  ({ index }) => {
    if (!index) {
      throw new Error("No index");
    }
    const sortParams = {
      sortRadial: dynoSortRadial,
      sortOrigin: dynoOrigin,
      sortDirection: dynoDirection,
      sortDepthBias: dynoDepthBias,
      sort360: dynoSort360,
    };

    const gsplat = dyno.readPackedSplat(dynoSplats, index);
    const metric = computeSortMetric({ gsplat, ...sortParams });
    const rgba8 = dyno.uintToRgba8(dyno.floatBitsToUint(metric));
    return { rgba8 };
  },
);

const defineComputeSortMetric = dyno.unindent(`
float computeSort(Gsplat gsplat, bool sortRadial, vec3 sortOrigin, vec3 sortDirection, float sortDepthBias, bool sort360) {
  if (!isGsplatActive(gsplat.flags)) {
    return INFINITY;
  }

  vec3 center = gsplat.center - sortOrigin;
  float biasedDepth = dot(center, sortDirection) + sortDepthBias;
  if (!sort360 && (biasedDepth <= 0.0)) {
    return INFINITY;
  }

  return sortRadial ? length(center) : biasedDepth;
}
`);

function computeSortMetric({
  gsplat,
  sortRadial,
  sortOrigin,
  sortDirection,
  sortDepthBias,
  sort360,
}: {
  gsplat: dyno.DynoVal<typeof dyno.Gsplat>;
  sortRadial: dyno.DynoVal<"bool">;
  sortOrigin: dyno.DynoVal<"vec3">;
  sortDirection: dyno.DynoVal<"vec3">;
  sortDepthBias: dyno.DynoVal<"float">;
  sort360: dyno.DynoVal<"bool">;
}) {
  return dyno.dyno({
    inTypes: {
      gsplat: dyno.Gsplat,
      sortRadial: "bool",
      sortOrigin: "vec3",
      sortDirection: "vec3",
      sortDepthBias: "float",
      sort360: "bool",
    },
    outTypes: { metric: "float" },
    globals: () => [dyno.defineGsplat, defineComputeSortMetric],
    inputs: {
      gsplat,
      sortRadial,
      sortOrigin,
      sortDirection,
      sortDepthBias,
      sort360,
    },
    statements: ({ inputs, outputs }) => {
      const {
        gsplat,
        sortRadial,
        sortOrigin,
        sortDirection,
        sortDepthBias,
        sort360,
      } = inputs;
      return dyno.unindentLines(`
        ${outputs.metric} = computeSort(${gsplat}, ${sortRadial}, ${sortOrigin}, ${sortDirection}, ${sortDepthBias}, ${sort360});
      `);
    },
  }).outputs.metric;
}
