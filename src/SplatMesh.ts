import * as THREE from "three";

import init_wasm, { raycast_splats } from "spark-internal-rs";
import {
  DynoPackedSplats,
  PackedSplats,
  type SplatEncoding,
} from "./PackedSplats";
import { DynoRgbaArray, type RgbaArray, TRgbaArray } from "./RgbaArray";
import type { GeneratorState } from "./SplatAccumulator";
import { SplatEdit, SplatEditSdf, SplatEdits } from "./SplatEdit";
import {
  type FrameUpdateContext,
  type GsplatModifier,
  type PrepareMultiContext,
  SplatGenerator,
  SplatTransformer,
} from "./SplatGenerator";
import type { SplatFileType } from "./SplatLoader";
import type { SplatSkinning } from "./SplatSkinning";
import {
  EXT_LN_SCALE_MAX,
  EXT_LN_SCALE_MIN,
  LN_SCALE_MAX,
  LN_SCALE_MIN,
} from "./defines";
import {
  DynoBool,
  DynoFloat,
  DynoIvec3,
  DynoUsampler2DArray,
  type DynoVal,
  DynoVec2,
  DynoVec4,
  Gsplat,
  combineGsplat,
  defineGsplat,
  dyno,
  dynoBlock,
  mul,
  readPackedSplat,
  splitGsplat,
  sub,
  unindent,
  unindentLines,
} from "./dyno";
import { getTextureSize } from "./utils";

export type SplatMeshOptions = {
  // URL to fetch a Gaussian splat file from(supports .ply, .splat, .ksplat,
  // .spz formats). (default: undefined)
  url?: string;
  // Raw bytes of a Gaussian splat file to decode directly instead of fetching
  // from URL. (default: undefined)
  fileBytes?: Uint8Array | ArrayBuffer;
  // Override the file type detection for formats that can't be reliably
  // auto-detected (.splat, .ksplat). (default: undefined auto-detects other
  // formats from file contents)
  fileType?: SplatFileType;
  // File name to use for type detection. (default: undefined)
  fileName?: string;
  // Use an existing PackedSplats object as the source instead of loading from
  // a file. Can be used to share a collection of Gsplats among multiple SplatMeshes
  // (default: undefined creates a new empty PackedSplats or decoded from a
  // data source above)
  packedSplats?: PackedSplats;
  // Reserve space for at least this many splats when constructing the mesh
  // initially. (default: determined by file)
  maxSplats?: number;
  // Callback function to programmatically create splats at initialization
  // in provided PackedSplats. (default: undefined)
  constructSplats?: (splats: PackedSplats) => Promise<void> | void;
  // Callback function that is called when mesh initialization is complete.
  // (default: undefined)
  onLoad?: (mesh: SplatMesh) => Promise<void> | void;
  // Controls whether SplatEdits have any effect on this mesh. (default: true)
  editable?: boolean;
  // Callback function that is called every frame to update the mesh.
  // Call mesh.updateVersion() if splats need to be regenerated due to some change.
  // Calling updateVersion() is not necessary for object transformations, recoloring,
  // or opacity adjustments as these are auto-detected. (default: undefined)
  onFrame?: ({
    mesh,
    time,
    deltaTime,
  }: { mesh: SplatMesh; time: number; deltaTime: number }) => void;
  // Gsplat modifier to apply in object-space before any transformations.
  // A GsplatModifier is a dyno shader-graph block that transforms an input
  // gsplat: DynoVal<Gsplat> to an output gsplat: DynoVal<Gsplat> with gsplat.center
  // coordinate in object-space. (default: undefined)
  objectModifier?: GsplatModifier;
  // Gsplat modifier to apply in world-space after transformations.
  // (default: undefined)
  worldModifier?: GsplatModifier;
  // Override the default splat encoding ranges for the PackedSplats.
  // (default: undefined)
  splatEncoding?: SplatEncoding;
};

export type LodMeta = {
  pixelSizes: [number, number][];
  boundCenter: THREE.Vector3;
  boundRadius: number;
};

export type SplatChunk = {
  packedSplats: PackedSplats;
  splatRgba?: RgbaArray;
  lodMeta?: LodMeta;
};

export type SplatMeshContext = {
  time: DynoFloat;
  deltaTime: DynoFloat;
  splats: DynoPackedSplats;
  splatRgba: DynoRgbaArray;
  objectToWorld: SplatTransformer;
  viewToWorld: SplatTransformer;
  worldToView: SplatTransformer;
  viewToObject: SplatTransformer;
  recolor: DynoVec4<THREE.Vector4>;
  rgbaDisplaceEdits: SplatEdits | null;
  shCounts: DynoIvec3<THREE.Vector3>;
  sh1Texture: DynoUsampler2DArray<"sh1", THREE.DataArrayTexture>;
  sh2Texture: DynoUsampler2DArray<"sh2", THREE.DataArrayTexture>;
  sh3Texture: DynoUsampler2DArray<"sh3", THREE.DataArrayTexture>;
  sh1MinMax: DynoVec2<"sh1MinMax", THREE.Vector2>;
  sh2MinMax: DynoVec2<"sh2MinMax", THREE.Vector2>;
  sh3MinMax: DynoVec2<"sh3MinMax", THREE.Vector2>;
  enableLod: DynoBool;
  pixelScale: DynoFloat;
  minPixelSize: DynoFloat;
};

export class SplatMesh extends SplatGenerator {
  // A Promise<SplatMesh> you can await to ensure fetching, parsing,
  // and initialization has completed
  initialized: Promise<SplatMesh>;
  // A boolean indicating whether initialization is complete
  isInitialized = false;

  // If you modify packedSplats you should set
  // splatMesh.packedSplats.needsUpdate = true to signal to Three.js that it
  // should re-upload the data to the underlying texture. Use this sparingly with
  // objects with smaller Gsplat counts as it requires a CPU-GPU data transfer for
  // each frame. Thousands to tens of thousands of Gsplats ir fine. (See hands.ts
  // for an example of rendering "Gsplat hands" in WebXR using this technique.)
  packedSplats: PackedSplats;

  // A THREE.Color that can be used to tint all splats in the mesh.
  // (default: new THREE.Color(1, 1, 1))
  recolor: THREE.Color = new THREE.Color(1, 1, 1);
  // Global opacity multiplier for all splats in the mesh. (default: 1)
  opacity = 1;

  // A SplatMeshContext consisting of useful scene and object dyno uniforms that can
  // be used to in the Gsplat processing pipeline, for example via objectModifier and
  // worldModifier. (created on construction)
  context: SplatMeshContext;
  onFrame?: ({
    mesh,
    time,
    deltaTime,
  }: { mesh: SplatMesh; time: number; deltaTime: number }) => void;

  objectModifier?: GsplatModifier;
  worldModifier?: GsplatModifier;

  // Optional SplatSkinning instance for animating splats with dual-quaternion
  // skeletal animation. (default: null)
  skinning: SplatSkinning | null = null;

  // Optional list of SplatEdits to apply to the mesh. If null, any SplatEdit
  // children in the scene graph will be added automatically. (default: null)
  edits: SplatEdit[] | null = null;
  editable: boolean;
  // Optional RgbaArray to overwrite splat RGBA values with custom values.
  // Useful for "baking" RGB and opacity edits into the SplatMesh. (default: null)
  splatRgba: RgbaArray | null = null;

  // Maximum Spherical Harmonics level to use. Call updateGenerator()
  // after changing. (default: 3)
  maxSh = 3;

  // Minimum opacity for raycast. (default: 0.1)
  minRaycastOpacity = 0.1;

  // Set to true to enable LoD modulation. (default: false)
  enableLod = false;
  // LoD scale adjustment factor (default: 1.0)
  lodScale = 1.0;
  // LoD tree and bounding metadata
  lodMeta: LodMeta | null = null;
  // Multi-splat chunks
  chunks: SplatChunk[] | null = null;

  constructor(options: SplatMeshOptions = {}) {
    super({
      update: (context) => this.update(context),
      prepareMulti: (context) => this.updateMulti(context),
    });

    this.packedSplats =
      options.packedSplats ??
      new PackedSplats({ splatEncoding: options.splatEncoding });
    this.numSplats = this.packedSplats.numSplats;
    this.editable = options.editable ?? true;
    this.onFrame = options.onFrame;

    this.context = {
      splats: new DynoPackedSplats(),
      objectToWorld: new SplatTransformer(),
      viewToWorld: new SplatTransformer(),
      worldToView: new SplatTransformer(),
      viewToObject: new SplatTransformer(),
      pixelScale: new DynoFloat({ value: 0 }),
      minPixelSize: new DynoFloat({ value: 0 }),
      recolor: new DynoVec4({
        value: new THREE.Vector4().setScalar(Number.NEGATIVE_INFINITY),
      }),
      time: new DynoFloat({ value: 0 }),
      deltaTime: new DynoFloat({ value: 0 }),
      splatRgba: new DynoRgbaArray(),
      rgbaDisplaceEdits: null,
      shCounts: new DynoIvec3({ value: new THREE.Vector3() }),
      sh1Texture: new DynoUsampler2DArray({
        value: getEmptySh1Texture(),
        key: "sh1",
      }),
      sh2Texture: new DynoUsampler2DArray({
        value: getEmptySh2Texture(),
        key: "sh2",
      }),
      sh3Texture: new DynoUsampler2DArray({
        value: getEmptySh3Texture(),
        key: "sh3",
      }),
      sh1MinMax: new DynoVec2({
        value: new THREE.Vector2(-1, 1),
        key: "sh1MinMax",
      }),
      sh2MinMax: new DynoVec2({
        value: new THREE.Vector2(-1, 1),
        key: "sh2MinMax",
      }),
      sh3MinMax: new DynoVec2({
        value: new THREE.Vector2(-1, 1),
        key: "sh3MinMax",
      }),
      enableLod: new DynoBool({ value: false }),
    };
    this.objectModifier = options.objectModifier;
    this.worldModifier = options.worldModifier;

    this.updateGenerator();

    if (
      options.url ||
      options.fileBytes ||
      options.constructSplats ||
      (options.packedSplats && !options.packedSplats.isInitialized)
    ) {
      // We need to initialize asynchronously given the options
      this.initialized = this.asyncInitialize(options).then(async () => {
        this.updateGenerator();

        this.isInitialized = true;
        if (options.onLoad) {
          const maybePromise = options.onLoad(this);
          if (maybePromise instanceof Promise) {
            await maybePromise;
          }
        }
        return this;
      });
    } else {
      this.isInitialized = true;
      this.initialized = Promise.resolve(this);
      if (options.onLoad) {
        const maybePromise = options.onLoad(this);
        // If onLoad returns a promise, wait for it to complete
        if (maybePromise instanceof Promise) {
          this.initialized = maybePromise.then(() => this);
        }
      }
    }
  }

  async asyncInitialize(options: SplatMeshOptions) {
    const {
      url,
      fileBytes,
      fileType,
      fileName,
      maxSplats,
      constructSplats,
      splatEncoding,
    } = options;
    if (url || fileBytes || constructSplats) {
      const packedSplatsOptions = {
        url,
        fileBytes,
        fileType,
        fileName,
        maxSplats,
        construct: constructSplats,
        splatEncoding,
      };
      this.packedSplats.reinitialize(packedSplatsOptions);
    }
    if (this.packedSplats) {
      await this.packedSplats.initialized;
      this.numSplats = this.packedSplats.numSplats;
      this.updateGenerator();
    }
  }

  static staticInitialized = SplatMesh.staticInitialize();
  static isStaticInitialized = false;

  static async staticInitialize() {
    await init_wasm();
    SplatMesh.isStaticInitialized = true;
  }

  getSplat(index: number) {
    return this.packedSplats.getSplat(index);
  }

  setSplat(
    index: number,
    center: THREE.Vector3,
    scales: THREE.Vector3,
    quaternion: THREE.Quaternion,
    opacity: number,
    color: THREE.Color,
    extra?: THREE.Vector2,
    lods?: THREE.Vector4,
  ) {
    this.packedSplats.setSplat(
      index,
      center,
      scales,
      quaternion,
      opacity,
      color,
      extra,
      lods,
    );
  }

  // Creates a new Gsplat with the provided parameters (all values in "float" space,
  // i.e. 0-1 for opacity and color) and adds it to the end of the packedSplats,
  // increasing numSplats by 1. If necessary, reallocates the buffer with an exponential
  // doubling strategy to fit the new data, so it's fairly efficient to just
  // pushSplat(...) each Gsplat you want to create in a loop.
  pushSplat(
    center: THREE.Vector3,
    scales: THREE.Vector3,
    quaternion: THREE.Quaternion,
    opacity: number,
    color: THREE.Color,
    extra?: THREE.Vector2,
    lods?: THREE.Vector4,
  ) {
    this.packedSplats.pushSplat(
      center,
      scales,
      quaternion,
      opacity,
      color,
      extra,
      lods,
    );
  }

  // This method iterates over all Gsplats in this instance's packedSplats,
  // invoking the provided callback with index: number in 0..=(this.numSplats-1) and
  // center: THREE.Vector3, scales: THREE.Vector3, quaternion: THREE.Quaternion,
  // opacity: number (0..1), and color: THREE.Color (rgb values in 0..1).
  // Note that the objects passed in as center etc. are the same for every callback
  // invocation: these objects are reused for efficiency. Changing these values has
  // no effect as they are decoded/unpacked copies of the underlying data. To update
  // the packedSplats, call .packedSplats.setSplat(index, center, scales,
  // quaternion, opacity, color).
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
      extra: THREE.Vector2,
      lods: THREE.Vector4,
    ) => void,
  ) {
    this.packedSplats.forEachSplat(callback);
  }

  // Call this when you are finished with the SplatMesh and want to free
  // any buffers it holds (via packedSplats).
  dispose() {
    this.packedSplats.dispose();
  }

  constructGenerator(context: SplatMeshContext) {
    this.generator = dynoBlock(
      { index: "int" },
      { gsplat: Gsplat },
      ({ index }) => {
        if (!index) {
          throw new Error("index is undefined");
        }
        // Read a Gsplat from the PackedSplats template
        let gsplat = readPackedSplat(context.splats, index);
        const { center } = splitGsplat(gsplat).outputs;

        // Calculate view direction in object space
        const viewCenterInObject = context.viewToObject.translate;
        const viewDelta = sub(center, viewCenterInObject);

        gsplat = maybeApplyLod(context, gsplat, viewDelta);
        gsplat = maybeApplySH(context, gsplat, viewDelta);
        gsplat = maybeInjectRgba(context, gsplat);

        if (this.skinning) {
          // Transform according to bones + skinning weights
          gsplat = this.skinning.modify(gsplat);
        }

        if (this.objectModifier) {
          // Inject object-space Gsplat modifier dyno
          gsplat = this.objectModifier.apply({ gsplat }).gsplat;
        }

        // Transform from object to world-space
        gsplat = context.objectToWorld.applyGsplat(gsplat);

        // Apply any global recoloring and opacity
        const recolorRgba = mul(
          this.context.recolor,
          splitGsplat(gsplat).outputs.rgba,
        );
        gsplat = combineGsplat({ gsplat, rgba: recolorRgba });

        if (this.context.rgbaDisplaceEdits) {
          // Apply RGBA edit layer SDFs
          gsplat = this.context.rgbaDisplaceEdits.modify(gsplat);
        }
        if (this.worldModifier) {
          // Inject world-space Gsplat modifier dyno
          gsplat = this.worldModifier.apply({ gsplat }).gsplat;
        }

        // We're done! Output resulting Gsplat
        return { gsplat };
      },
    );
  }

  // Call this whenever something changes in the Gsplat processing pipeline,
  // for example changing maxSh or updating objectModifier or worldModifier.
  // Compiled generators are cached for efficiency and re-use when the same
  // pipeline structure emerges after successive changes.
  updateGenerator() {
    this.constructGenerator(this.context);
  }

  updateSplatContext(chunk?: SplatChunk) {
    const packedSplats = chunk?.packedSplats ?? this.packedSplats;
    const splatRgba = chunk?.splatRgba ?? this.splatRgba;

    this.context.splats.packedSplats = packedSplats;
    this.context.splatRgba.rgbaArray = splatRgba ?? undefined;

    if (this.maxSh >= 1 && packedSplats.extra.sh1) {
      const { sh1Texture, sh2Texture, sh3Texture } =
        this.ensureShTextures(packedSplats);
      this.context.shCounts.value.set(
        sh1Texture ? packedSplats.numSplats : 0,
        sh2Texture && this.maxSh >= 2 ? packedSplats.numSplats : 0,
        sh3Texture && this.maxSh >= 3 ? packedSplats.numSplats : 0,
      );
      this.context.sh1Texture.value = sh1Texture?.value ?? getEmptySh1Texture();
      this.context.sh2Texture.value = sh2Texture?.value ?? getEmptySh2Texture();
      this.context.sh3Texture.value = sh3Texture?.value ?? getEmptySh3Texture();
      const { sh1Min, sh1Max, sh2Min, sh2Max, sh3Min, sh3Max } =
        packedSplats.splatEncoding ?? {};
      this.context.sh1MinMax.value.set(sh1Min ?? -1, sh1Max ?? 1);
      this.context.sh2MinMax.value.set(sh2Min ?? -1, sh2Max ?? 1);
      this.context.sh3MinMax.value.set(sh3Min ?? -1, sh3Max ?? 1);
    } else {
      this.context.shCounts.value.set(0, 0, 0);
    }

    this.context.enableLod.value =
      this.enableLod && (packedSplats.splatEncoding?.extended ?? false);
  }

  // This is called automatically by SparkRenderer and you should not have to
  // call it. It updates parameters for the generated pipeline and calls
  // updateGenerator() if the pipeline needs to change.
  update({
    time,
    deltaTime,
    viewToWorld,
    camera,
    renderSize,
    globalEdits,
    sortState,
    lastState,
    newState,
  }: FrameUpdateContext) {
    const context = this.context;
    context.time.value = time;
    context.deltaTime.value = deltaTime;

    this.updateSplatContext();

    let updated = context.objectToWorld.update(this);
    if (context.viewToWorld.updateFromMatrix(viewToWorld)) {
      updated = true;
    }

    const worldToView = viewToWorld.clone().invert();
    if (context.worldToView.updateFromMatrix(worldToView)) {
      updated = true;
    }

    const worldToObject = this.matrixWorld.clone().invert();
    const viewToObjectMatrix = worldToObject.multiply(viewToWorld);
    if (context.viewToObject.updateFromMatrix(viewToObjectMatrix)) {
      updated = true;
    }

    const recolor = context.recolor.value;
    if (
      recolor.x !== this.recolor.r ||
      recolor.y !== this.recolor.g ||
      recolor.z !== this.recolor.b ||
      recolor.w !== this.opacity
    ) {
      recolor.set(this.recolor.r, this.recolor.g, this.recolor.b, this.opacity);
      updated = true;
    }

    const edits = this.editable ? (this.edits ?? []).concat(globalEdits) : [];
    if (this.editable && !this.edits) {
      // If we haven't set any explicit edits, add any child SplatEdits
      this.traverseVisible((node) => {
        if (node instanceof SplatEdit) {
          edits.push(node);
        }
      });
    }

    edits.sort((a, b) => a.ordering - b.ordering);
    const editsSdfs = edits.map((edit) => {
      if (edit.sdfs != null) {
        return { edit, sdfs: edit.sdfs };
      }
      const sdfs: SplatEditSdf[] = [];
      edit.traverseVisible((node) => {
        if (node instanceof SplatEditSdf) {
          sdfs.push(node);
        }
      });
      return { edit, sdfs };
    });

    if (editsSdfs.length > 0 && !context.rgbaDisplaceEdits) {
      const edits = editsSdfs.length;
      const sdfs = editsSdfs.reduce(
        (total, edit) => total + edit.sdfs.length,
        0,
      );
      context.rgbaDisplaceEdits = new SplatEdits({
        maxEdits: edits,
        maxSdfs: sdfs,
      });
      this.updateGenerator();
    }
    if (context.rgbaDisplaceEdits) {
      const editResult = context.rgbaDisplaceEdits.update(editsSdfs);
      updated ||= editResult.updated;
      if (editResult.dynoUpdated) {
        this.updateGenerator();
      }
    }

    let pixelScale = 0.0;
    if (renderSize) {
      pixelScale =
        camera instanceof THREE.PerspectiveCamera
          ? (2.0 * Math.tan((0.5 * camera.fov * Math.PI) / 180.0)) /
            renderSize.y
          : 0.0;
      pixelScale *= this.lodScale;
    }

    if (this.context.pixelScale.value !== pixelScale) {
      this.context.pixelScale.value = pixelScale;
      if (this.enableLod) {
        updated = true;
      }
    }

    let numSplats = this.packedSplats.numSplats;
    if (this.enableLod && this.lodMeta) {
      const distance = context.viewToObject.translate.value.distanceTo(
        this.lodMeta.boundCenter,
      );
      const closest = Math.max(0, distance - this.lodMeta.boundRadius);
      const pixelSize = closest * pixelScale;
      const cut = searchPixelSizes(this.lodMeta.pixelSizes, pixelSize);

      const cutSplats =
        cut < this.lodMeta.pixelSizes.length
          ? this.lodMeta.pixelSizes[cut][1]
          : Number.POSITIVE_INFINITY;
      numSplats = Math.min(numSplats, cutSplats);

      let minPixelSize =
        cut < this.lodMeta.pixelSizes.length
          ? this.lodMeta.pixelSizes[cut][0]
          : 0.0;
      (newState as { minPixelSize: number }).minPixelSize = minPixelSize;
      const sortPixelSize = (sortState as { minPixelSize?: number } | undefined)
        ?.minPixelSize;
      minPixelSize = Math.max(
        minPixelSize,
        sortPixelSize ?? Number.POSITIVE_INFINITY,
      );
      context.minPixelSize.value = minPixelSize;
    }

    if (this.numSplats !== numSplats) {
      this.numSplats = numSplats;
      updated = true;
    }

    if (this.chunks) {
      if (!this.multiSplats) {
        this.multiSplats = new Map();
      } else {
        this.multiSplats.clear();
      }
      for (const chunk of this.chunks) {
        let chunkSplats = chunk.packedSplats.numSplats;
        if (this.enableLod && chunk.lodMeta) {
          const distance = context.viewToObject.translate.value.distanceTo(
            chunk.lodMeta.boundCenter,
          );
          const closest = Math.max(0, distance - chunk.lodMeta.boundRadius);
          const pixelSize = closest * pixelScale;
          const cut = searchPixelSizes(chunk.lodMeta.pixelSizes, pixelSize);
          const cutSplats =
            cut < chunk.lodMeta.pixelSizes.length
              ? chunk.lodMeta.pixelSizes[cut][1]
              : Number.POSITIVE_INFINITY;
          chunkSplats = Math.min(chunkSplats, cutSplats);
        }
        this.multiSplats.set(chunk, chunkSplats);
      }
    } else {
      this.multiSplats = undefined;
    }

    if (updated) {
      this.updateVersion();
    }

    this.onFrame?.({ mesh: this, time, deltaTime });
  }

  updateMulti({
    object,
    multi,
    sortState,
    lastState,
    newState,
  }: PrepareMultiContext) {
    const chunk = multi as SplatChunk;
    this.updateSplatContext(chunk);

    if (this.enableLod && chunk.lodMeta) {
      const distance = this.context.viewToObject.translate.value.distanceTo(
        chunk.lodMeta.boundCenter,
      );
      const closest = Math.max(0, distance - chunk.lodMeta.boundRadius);
      const pixelSize = closest * this.context.pixelScale.value;
      const cut = searchPixelSizes(chunk.lodMeta.pixelSizes, pixelSize);

      let minPixelSize =
        cut < chunk.lodMeta.pixelSizes.length
          ? chunk.lodMeta.pixelSizes[cut][0]
          : 0.0;
      (newState as { minPixelSize: number }).minPixelSize = minPixelSize;
      const sortPixelSize = (sortState as { minPixelSize?: number } | undefined)
        ?.minPixelSize;
      minPixelSize = Math.max(
        minPixelSize,
        sortPixelSize ?? Number.POSITIVE_INFINITY,
      );
      this.context.minPixelSize.value = minPixelSize;
    }
  }

  // This method conforms to the standard THREE.Raycaster API, performing object-ray
  // intersections using this method to populate the provided intersects[] array
  // with each intersection point.
  raycast(
    raycaster: THREE.Raycaster,
    intersects: {
      distance: number;
      point: THREE.Vector3;
      object: THREE.Object3D;
    }[],
  ) {
    if (!this.packedSplats.packedArray || !this.packedSplats.numSplats) {
      return;
    }

    const { near, far, ray } = raycaster;
    const worldToMesh = this.matrixWorld.clone().invert();
    const worldToMeshRot = new THREE.Matrix3().setFromMatrix4(worldToMesh);
    const origin = ray.origin.clone().applyMatrix4(worldToMesh);
    const direction = ray.direction.clone().applyMatrix3(worldToMeshRot);
    const scales = new THREE.Vector3();
    worldToMesh.decompose(new THREE.Vector3(), new THREE.Quaternion(), scales);
    const scale = (scales.x * scales.y * scales.z) ** (1.0 / 3.0);

    const RAYCAST_ELLIPSOID = true;
    const lnScaleMin =
      this.packedSplats.splatEncoding?.lnScaleMin ??
      (this.packedSplats.splatEncoding?.extended
        ? EXT_LN_SCALE_MIN
        : LN_SCALE_MIN);
    const lnScaleMax =
      this.packedSplats.splatEncoding?.lnScaleMax ??
      (this.packedSplats.splatEncoding?.extended
        ? EXT_LN_SCALE_MAX
        : LN_SCALE_MAX);

    const distances = raycast_splats(
      origin.x,
      origin.y,
      origin.z,
      direction.x,
      direction.y,
      direction.z,
      near,
      far,
      this.packedSplats.numSplats,
      Array.isArray(this.packedSplats.packedArray)
        ? this.packedSplats.packedArray[0]
        : this.packedSplats.packedArray,
      Array.isArray(this.packedSplats.packedArray)
        ? this.packedSplats.packedArray[1]
        : undefined,
      RAYCAST_ELLIPSOID,
      lnScaleMin,
      lnScaleMax,
      this.minRaycastOpacity,
    );

    for (const distance of distances) {
      const point = ray.direction
        .clone()
        .multiplyScalar(distance)
        .add(ray.origin);
      intersects.push({
        distance,
        point,
        object: this,
      });
    }
  }

  private ensureShTextures(packedSplats: PackedSplats): {
    sh1Texture?: DynoUsampler2DArray<"sh1", THREE.DataArrayTexture>;
    sh2Texture?: DynoUsampler2DArray<"sh2", THREE.DataArrayTexture>;
    sh3Texture?: DynoUsampler2DArray<"sh3", THREE.DataArrayTexture>;
  } {
    // Ensure we have textures for SH1..SH3 if we have data
    if (!packedSplats.extra.sh1) {
      return {};
    }

    let sh1Texture = packedSplats.extra.sh1Texture as
      | DynoUsampler2DArray<"sh1", THREE.DataArrayTexture>
      | undefined;
    if (!sh1Texture) {
      let sh1 = packedSplats.extra.sh1 as Uint32Array<ArrayBuffer>;
      const { width, height, depth, maxSplats } = getTextureSize(
        sh1.length / 2,
      );
      if (sh1.length < maxSplats * 2) {
        const newSh1 = new Uint32Array(maxSplats * 2);
        newSh1.set(sh1);
        packedSplats.extra.sh1 = newSh1;
        sh1 = newSh1;
      }

      const texture = new THREE.DataArrayTexture(sh1, width, height, depth);
      texture.format = THREE.RGIntegerFormat;
      texture.type = THREE.UnsignedIntType;
      texture.internalFormat = "RG32UI";
      texture.needsUpdate = true;

      sh1Texture = new DynoUsampler2DArray({
        value: texture,
        key: "sh1",
      });
      packedSplats.extra.sh1Texture = sh1Texture;
    }

    if (!packedSplats.extra.sh2) {
      return { sh1Texture };
    }

    let sh2Texture = packedSplats.extra.sh2Texture as
      | DynoUsampler2DArray<"sh2", THREE.DataArrayTexture>
      | undefined;
    if (!sh2Texture) {
      let sh2 = packedSplats.extra.sh2 as Uint32Array<ArrayBuffer>;
      const { width, height, depth, maxSplats } = getTextureSize(
        sh2.length / 4,
      );
      if (sh2.length < maxSplats * 4) {
        const newSh2 = new Uint32Array(maxSplats * 4);
        newSh2.set(sh2);
        packedSplats.extra.sh2 = newSh2;
        sh2 = newSh2;
      }

      const texture = new THREE.DataArrayTexture(sh2, width, height, depth);
      texture.format = THREE.RGBAIntegerFormat;
      texture.type = THREE.UnsignedIntType;
      texture.internalFormat = "RGBA32UI";
      texture.needsUpdate = true;

      sh2Texture = new DynoUsampler2DArray({
        value: texture,
        key: "sh2",
      });
      packedSplats.extra.sh2Texture = sh2Texture;
    }

    if (!packedSplats.extra.sh3) {
      return { sh1Texture, sh2Texture };
    }

    let sh3Texture = packedSplats.extra.sh3Texture as
      | DynoUsampler2DArray<"sh3", THREE.DataArrayTexture>
      | undefined;
    if (!sh3Texture) {
      let sh3 = packedSplats.extra.sh3 as Uint32Array<ArrayBuffer>;
      const { width, height, depth, maxSplats } = getTextureSize(
        sh3.length / 4,
      );
      if (sh3.length < maxSplats * 4) {
        const newSh3 = new Uint32Array(maxSplats * 4);
        newSh3.set(sh3);
        packedSplats.extra.sh3 = newSh3;
        sh3 = newSh3;
      }

      const texture = new THREE.DataArrayTexture(sh3, width, height, depth);
      texture.format = THREE.RGBAIntegerFormat;
      texture.type = THREE.UnsignedIntType;
      texture.internalFormat = "RGBA32UI";
      texture.needsUpdate = true;

      sh3Texture = new DynoUsampler2DArray({
        value: texture,
        key: "sh3",
      });
      packedSplats.extra.sh3Texture = sh3Texture;
    }

    return { sh1Texture, sh2Texture, sh3Texture };
  }
}

const defineEvaluateSH1 = unindent(`
  vec3 evaluateSH1(usampler2DArray sh1, int index, vec3 viewDir) {
    // Extract sint7 values packed into 2 x uint32
    uvec2 packed = texelFetch(sh1, splatTexCoord(index), 0).rg;
    vec3 sh1_0 = vec3(ivec3(
      int(packed.x << 25u) >> 25,
      int(packed.x << 18u) >> 25,
      int(packed.x << 11u) >> 25
    )) / 63.0;
    vec3 sh1_1 = vec3(ivec3(
      int(packed.x << 4u) >> 25,
      int((packed.x >> 3u) | (packed.y << 29u)) >> 25,
      int(packed.y << 22u) >> 25
    )) / 63.0;
    vec3 sh1_2 = vec3(ivec3(
      int(packed.y << 15u) >> 25,
      int(packed.y << 8u) >> 25,
      int(packed.y << 1u) >> 25
    )) / 63.0;

    return sh1_0 * (-0.4886025 * viewDir.y)
      + sh1_1 * (0.4886025 * viewDir.z)
      + sh1_2 * (-0.4886025 * viewDir.x);
  }
`);

const defineEvaluateSH2 = unindent(`
  vec3 evaluateSH2(usampler2DArray sh2, int index, vec3 viewDir) {
    // Extract sint8 values packed into 4 x uint32
    uvec4 packed = texelFetch(sh2, splatTexCoord(index), 0);
    vec3 sh2_0 = vec3(ivec3(
      int(packed.x << 24u) >> 24,
      int(packed.x << 16u) >> 24,
      int(packed.x << 8u) >> 24
    )) / 127.0;
    vec3 sh2_1 = vec3(ivec3(
      int(packed.x) >> 24,
      int(packed.y << 24u) >> 24,
      int(packed.y << 16u) >> 24
    )) / 127.0;
    vec3 sh2_2 = vec3(ivec3(
      int(packed.y << 8u) >> 24,
      int(packed.y) >> 24,
      int(packed.z << 24u) >> 24
    )) / 127.0;
    vec3 sh2_3 = vec3(ivec3(
      int(packed.z << 16u) >> 24,
      int(packed.z << 8u) >> 24,
      int(packed.z) >> 24
    )) / 127.0;
    vec3 sh2_4 = vec3(ivec3(
      int(packed.w << 24u) >> 24,
      int(packed.w << 16u) >> 24,
      int(packed.w << 8u) >> 24
    )) / 127.0;

    return sh2_0 * (1.0925484 * viewDir.x * viewDir.y)
      + sh2_1 * (-1.0925484 * viewDir.y * viewDir.z)
      + sh2_2 * (0.3153915 * (2.0 * viewDir.z * viewDir.z - viewDir.x * viewDir.x - viewDir.y * viewDir.y))
      + sh2_3 * (-1.0925484 * viewDir.x * viewDir.z)
      + sh2_4 * (0.5462742 * (viewDir.x * viewDir.x - viewDir.y * viewDir.y));
  }
`);

const defineEvaluateSH3 = unindent(`
  vec3 evaluateSH3(usampler2DArray sh3, int index, vec3 viewDir) {
    // Extract sint6 values packed into 4 x uint32
    uvec4 packed = texelFetch(sh3, splatTexCoord(index), 0);
    vec3 sh3_0 = vec3(ivec3(
      int(packed.x << 26u) >> 26,
      int(packed.x << 20u) >> 26,
      int(packed.x << 14u) >> 26
    )) / 31.0;
    vec3 sh3_1 = vec3(ivec3(
      int(packed.x << 8u) >> 26,
      int(packed.x << 2u) >> 26,
      int((packed.x >> 4u) | (packed.y << 28u)) >> 26
    )) / 31.0;
    vec3 sh3_2 = vec3(ivec3(
      int(packed.y << 22u) >> 26,
      int(packed.y << 16u) >> 26,
      int(packed.y << 10u) >> 26
    )) / 31.0;
    vec3 sh3_3 = vec3(ivec3(
      int(packed.y << 4u) >> 26,
      int((packed.y >> 2u) | (packed.z << 30u)) >> 26,
      int(packed.z << 24u) >> 26
    )) / 31.0;
    vec3 sh3_4 = vec3(ivec3(
      int(packed.z << 18u) >> 26,
      int(packed.z << 12u) >> 26,
      int(packed.z << 6u) >> 26
    )) / 31.0;
    vec3 sh3_5 = vec3(ivec3(
      int(packed.z) >> 26,
      int(packed.w << 26u) >> 26,
      int(packed.w << 20u) >> 26
    )) / 31.0;
    vec3 sh3_6 = vec3(ivec3(
      int(packed.w << 14u) >> 26,
      int(packed.w << 8u) >> 26,
      int(packed.w << 2u) >> 26
    )) / 31.0;

    float xx = viewDir.x * viewDir.x;
    float yy = viewDir.y * viewDir.y;
    float zz = viewDir.z * viewDir.z;
    float xy = viewDir.x * viewDir.y;
    float yz = viewDir.y * viewDir.z;
    float zx = viewDir.z * viewDir.x;

    return sh3_0 * (-0.5900436 * viewDir.y * (3.0 * xx - yy))
      + sh3_1 * (2.8906114 * xy * viewDir.z) +
      + sh3_2 * (-0.4570458 * viewDir.y * (4.0 * zz - xx - yy))
      + sh3_3 * (0.3731763 * viewDir.z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
      + sh3_4 * (-0.4570458 * viewDir.x * (4.0 * zz - xx - yy))
      + sh3_5 * (1.4453057 * viewDir.z * (xx - yy))
      + sh3_6 * (-0.5900436 * viewDir.x * (xx - 3.0 * yy));
  }
`);

const defineRescaleSH = unindent(`
  vec3 rescaleSH(vec3 rgb, vec2 minMax) {
    float mid = 0.5 * (minMax.x + minMax.y);
    float scale = 0.5 * (minMax.y - minMax.x);
    return mid + scale * rgb;
  }
`);

function maybeApplySH(
  context: SplatMeshContext,
  gsplat: DynoVal<typeof Gsplat>,
  viewDelta: DynoVal<"vec3">,
) {
  const {
    shCounts,
    sh1Texture,
    sh2Texture,
    sh3Texture,
    sh1MinMax,
    sh2MinMax,
    sh3MinMax,
  } = context;
  return dyno({
    inTypes: {
      gsplat: Gsplat,
      viewDelta: "vec3",
      shCounts: "ivec3",
      sh1Texture: "usampler2DArray",
      sh2Texture: "usampler2DArray",
      sh3Texture: "usampler2DArray",
      sh1MinMax: "vec2",
      sh2MinMax: "vec2",
      sh3MinMax: "vec2",
    },
    outTypes: { gsplat: Gsplat },
    inputs: {
      gsplat,
      viewDelta,
      shCounts,
      sh1Texture,
      sh2Texture,
      sh3Texture,
      sh1MinMax,
      sh2MinMax,
      sh3MinMax,
    },
    globals: () => [
      defineGsplat,
      defineEvaluateSH1,
      defineEvaluateSH2,
      defineEvaluateSH3,
      defineRescaleSH,
    ],
    statements: ({ inputs, outputs }) =>
      unindentLines(`
        ${outputs.gsplat} = ${inputs.gsplat};
        if (isGsplatActive(${outputs.gsplat}.flags) && any(equal(${inputs.shCounts}, ivec3(0)))) {
          int index = ${outputs.gsplat}.index;
          vec3 viewDir = normalize(${inputs.viewDelta});

          if (index < ${inputs.shCounts}.x) {
            vec3 sh1Snorm = evaluateSH1(${inputs.sh1Texture}, index, viewDir);
            ${outputs.gsplat}.rgba.rgb += rescaleSH(sh1Snorm, ${inputs.sh1MinMax});
          }
          if (index < ${inputs.shCounts}.y) {
            vec3 sh2Snorm = evaluateSH2(${inputs.sh2Texture}, index, viewDir);
            ${outputs.gsplat}.rgba.rgb += rescaleSH(sh2Snorm, ${inputs.sh2MinMax});
          }
          if (index < ${inputs.shCounts}.z) {
            vec3 sh3Snorm = evaluateSH3(${inputs.sh3Texture}, index, viewDir);
            ${outputs.gsplat}.rgba.rgb += rescaleSH(sh3Snorm, ${inputs.sh3MinMax});
          }
        }
      `),
  }).outputs.gsplat;
}

const emptyShTextures: THREE.DataArrayTexture[] = [];

function makeEmptyShTexture(n: number) {
  const texture = new THREE.DataArrayTexture(
    new Uint32Array(n === 1 ? 2 : 4),
    1,
    1,
    1,
  );
  texture.format = n === 1 ? THREE.RGIntegerFormat : THREE.RGBAIntegerFormat;
  texture.type = THREE.UnsignedIntType;
  texture.internalFormat = n === 1 ? "RG32UI" : "RGBA32UI";
  texture.needsUpdate = true;
  return texture;
}

function getEmptyShTextures() {
  if (emptyShTextures.length === 0) {
    emptyShTextures.push(
      makeEmptyShTexture(1),
      makeEmptyShTexture(2),
      makeEmptyShTexture(3),
    );
  }
  return emptyShTextures;
}

function getEmptySh1Texture() {
  return getEmptyShTextures()[0];
}

function getEmptySh2Texture() {
  return getEmptyShTextures()[1];
}

function getEmptySh3Texture() {
  return getEmptyShTextures()[2];
}

const defineModulateLod = unindent(`
  float modulateLod(float pixelSize, vec4 lods) {
    if (pixelSize <= 0.0) {
      return 1.0;
    }
    if ((lods.w > 0.0) && (pixelSize >= lods.w)) {
      return 0.0;
    }
    if (pixelSize <= lods.x) {
      return 0.0;
    }
    if ((lods.z > 0.0) && (pixelSize > lods.z)) {
      vec2 logs = log(lods.wz);
      return (log(pixelSize) - logs.x) / (logs.y - logs.x);
    }
    if (pixelSize < lods.y) {
      vec2 logs = log(lods.xy);
      return (log(pixelSize) - logs.x) / (logs.y - logs.x);
    }
    return 1.0;
  }
`);

function maybeApplyLod(
  context: SplatMeshContext,
  gsplat: DynoVal<typeof Gsplat>,
  viewDelta: DynoVal<"vec3">,
) {
  const { enableLod, pixelScale, minPixelSize } = context;
  return dyno({
    inTypes: {
      gsplat: Gsplat,
      viewDelta: "vec3",
      enableLod: "bool",
      pixelScale: "float",
      minPixelSize: "float",
    },
    outTypes: { gsplat: Gsplat },
    inputs: { gsplat, viewDelta, enableLod, pixelScale, minPixelSize },
    globals: () => [defineGsplat, defineModulateLod],
    statements: ({ inputs, outputs }) =>
      unindentLines(`
        ${outputs.gsplat} = ${inputs.gsplat};
        if (${inputs.enableLod} && isGsplatActive(${outputs.gsplat}.flags)) {
          float distance = length(${inputs.viewDelta});
          float pixelSize = max(${inputs.minPixelSize}, distance * ${inputs.pixelScale});
          ${outputs.gsplat}.rgba.a *= modulateLod(pixelSize, ${outputs.gsplat}.lods);
        }
      `),
  }).outputs.gsplat;
}

function maybeInjectRgba(
  context: SplatMeshContext,
  gsplat: DynoVal<typeof Gsplat>,
) {
  const { splatRgba } = context;
  return dyno({
    inTypes: { gsplat: Gsplat, splatRgba: TRgbaArray },
    outTypes: { gsplat: Gsplat },
    inputs: { gsplat, splatRgba },
    globals: () => [defineGsplat],
    statements: ({ inputs, outputs }) =>
      unindentLines(`
        ${outputs.gsplat} = ${inputs.gsplat};
        if (isGsplatActive(${outputs.gsplat}.flags) && (${outputs.gsplat}.index < ${inputs.splatRgba}.count)) {
          ${outputs.gsplat}.rgba = texelFetch(${inputs.splatRgba}.texture, splatTexCoord(${outputs.gsplat}.index), 0);
        }
      `),
  }).outputs.gsplat;
}

function searchPixelSizes(pixelSizes: [number, number][], pixelSize: number) {
  let lo = 0;
  let hi = pixelSizes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pixelSizes[mid][0] <= pixelSize) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
