import { Gunzip, gunzipSync } from "fflate";
import * as THREE from "three";
import type { SplatEncoding } from "./PackedSplats";
import {
  computeMaxSplats,
  encodeSh1Rgb,
  encodeSh2Rgb,
  encodeSh3Rgb,
  fromHalf,
  setPackedSplatCenter,
  setPackedSplatLods,
  setPackedSplatOpacity,
  setPackedSplatQuat,
  setPackedSplatRgb,
  setPackedSplatScales,
} from "./utils";

export type RadMeta = {
  version: number;
  type: RadType;
  count: number;
  antialias?: boolean;
  payloads: RadPayload[];
  lodMeta?: RadLodMeta;
};

export enum RadType {
  Gsplat = "gsplat",
}

export type RadPayload = {
  bytes: number;
  property: RadProperty;
  encoding: RadEncoding;
  base: number;
  count: number;
  components?: number;
  compression?: RadCompression;
  mins?: number[];
  maxs?: number[];
};

export enum RadProperty {
  Center = "center",
  Alpha = "alpha",
  Rgb = "rgb",
  Scales = "scales",
  Orientation = "orientation",
  LodScales = "lodScales",
}

export enum RadEncoding {
  Float16ByteOrder = "float16ByteOrder",
  Uint8 = "uint8",
  LogUint8 = "logUint8",
  Oct88R8 = "oct88R8",
}

export enum RadCompression {
  Gzip = "gzip",
}

export type RadLodMeta = {
  pixelSizes: [number, number][];
  boundCenter: [number, number, number];
  boundRadius: number;
};

export class RadReader {
  fileBytes: Uint8Array;
  meta: RadMeta;
  numSplats: number;
  payloads: { meta: RadPayload; bytes: Uint8Array }[];

  constructor({ fileBytes }: { fileBytes: Uint8Array }) {
    this.fileBytes = fileBytes;
    const headView = new DataView(fileBytes.buffer, 0, 8);
    if (headView.getUint32(0, true) !== 0x30444152) {
      throw new Error("Invalid RAD file");
    }
    const metaSize = headView.getUint32(4, true);
    const metaBytes = fileBytes.slice(8, 8 + metaSize);

    const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as RadMeta;
    this.meta = meta;
    if (meta.version !== 1) {
      throw new Error(`Unsupported RAD version: ${meta.version}`);
    }
    if (meta.type !== RadType.Gsplat) {
      throw new Error(`Unsupported RAD type: ${meta.type}`);
    }
    this.numSplats = meta.count;

    const payloads: { meta: RadPayload; bytes: Uint8Array }[] = [];
    let offset = 8 + roundUp8(metaSize);
    for (const payloadMeta of meta.payloads) {
      const headView = new DataView(fileBytes.buffer, offset, 8);
      const bytes = Number(headView.getBigUint64(0, true));
      if (bytes !== payloadMeta.bytes) {
        throw new Error(
          `Mismatched RAD payload size: ${bytes} !== ${payloadMeta.bytes}`,
        );
      }
      offset += 8;

      payloads.push({
        meta: payloadMeta,
        bytes: fileBytes.slice(offset, offset + bytes),
      });
      offset += roundUp8(bytes);
    }
    this.payloads = payloads;

    const tailView = new DataView(fileBytes.buffer, offset, 8);
    if (tailView.getBigUint64(0, true) !== BigInt(0)) {
      throw new Error("Invalid RAD file");
    }
  }

  parseSplats({
    center,
    alpha,
    rgb,
    scales,
    quat,
    lodScales,
    sh,
    setEncoding,
  }: {
    center?: (index: number, x: number, y: number, z: number) => void;
    alpha?: (index: number, alpha: number) => void;
    rgb?: (index: number, r: number, g: number, b: number) => void;
    scales?: (
      index: number,
      scaleX: number,
      scaleY: number,
      scaleZ: number,
    ) => void;
    quat?: (
      index: number,
      quatX: number,
      quatY: number,
      quatZ: number,
      quatW: number,
    ) => void;
    lodScales?: (
      index: number,
      lodMin: number,
      lodLow: number,
      lodHigh: number,
      lodMax: number,
    ) => void;
    sh?: (
      index: number,
      sh1: Float32Array,
      sh2?: Float32Array,
      sh3?: Float32Array,
    ) => void;
    setEncoding?: ({
      rgbMin,
      rgbMax,
      lnScaleMin,
      lnScaleMax,
      sh1Min,
      sh1Max,
      sh2Min,
      sh2Max,
      sh3Min,
      sh3Max,
    }: {
      rgbMin?: number;
      rgbMax?: number;
      lnScaleMin?: number;
      lnScaleMax?: number;
      sh1Min?: number;
      sh1Max?: number;
      sh2Min?: number;
      sh2Max?: number;
      sh3Min?: number;
      sh3Max?: number;
    }) => void;
  }) {
    for (let { meta, bytes } of this.payloads) {
      const { property, encoding, base, count, compression } = meta;
      const callback = (
        property === RadProperty.Center
          ? center
          : property === RadProperty.Alpha
            ? alpha
            : property === RadProperty.Rgb
              ? rgb
              : property === RadProperty.Scales
                ? scales
                : property === RadProperty.Orientation
                  ? quat
                  : property === RadProperty.LodScales
                    ? lodScales
                    : undefined
      ) as ((...args: unknown[]) => void) | undefined;
      if (!callback) {
        continue;
      }

      if (compression === RadCompression.Gzip) {
        bytes = gunzipSync(bytes);
      }

      const components = meta.components ?? 1;

      const mins: number[] = meta.mins ?? new Array();
      while (mins.length < (components ?? 1)) {
        mins.push(mins[0] ?? 0);
      }
      const maxs: number[] = meta.maxs ?? new Array();
      while (maxs.length < components) {
        maxs.push(maxs[0] ?? 1);
      }

      switch (encoding) {
        case RadEncoding.Float16ByteOrder: {
          const expectedBytes = 2 * count * components;
          if (bytes.length !== expectedBytes) {
            throw new Error(
              `Mismatched RAD encoding: ${bytes.length} !== ${expectedBytes}`,
            );
          }

          const args = new Array(1 + components).fill(0);
          for (let i = 0; i < count; ++i) {
            args[0] = base + i;
            const planeSize = components * count;
            for (let d = 0; d < components; ++d) {
              const offset = d * count + i;
              const u16 = bytes[offset] | (bytes[planeSize + offset] << 8);
              args[1 + d] = fromHalf(u16);
            }
            callback.apply(null, args);
          }
          break;
        }
        case RadEncoding.Uint8: {
          const expectedBytes = count * components;
          if (bytes.length !== expectedBytes) {
            throw new Error(
              `Mismatched RAD encoding: ${bytes.length} !== ${expectedBytes}`,
            );
          }

          if (property === RadProperty.Rgb) {
            setEncoding?.({ rgbMin: mins[0], rgbMax: maxs[0] });
          }

          const args = new Array(1 + components).fill(0);
          for (let i = 0; i < count; ++i) {
            args[0] = base + i;
            for (let d = 0; d < components; ++d) {
              args[1 + d] =
                (bytes[d * count + i] / 255) * (maxs[d] - mins[d]) + mins[d];
            }
            callback.apply(null, args);
          }
          break;
        }
        case RadEncoding.LogUint8: {
          const expectedBytes = count * components;
          if (bytes.length !== expectedBytes) {
            throw new Error(
              `Mismatched RAD encoding: ${bytes.length} !== ${expectedBytes}`,
            );
          }

          if (components === 3 && property === RadProperty.Scales) {
            setEncoding?.({ lnScaleMin: mins[0], lnScaleMax: maxs[0] });
          }

          const args = new Array(1 + components).fill(0);
          for (let i = 0; i < count; ++i) {
            args[0] = base + i;
            for (let d = 0; d < components; ++d) {
              const u8 = bytes[d * count + i];
              args[1 + d] =
                u8 === 0
                  ? 0
                  : Math.exp(((u8 - 1) / 254) * (maxs[d] - mins[d]) + mins[d]);
            }
            callback.apply(null, args);
          }
          break;
        }
        case RadEncoding.Oct88R8: {
          const expectedBytes = 3 * count;
          if (bytes.length !== expectedBytes) {
            throw new Error(
              `Mismatched RAD encoding: ${bytes.length} !== ${expectedBytes}`,
            );
          }

          const args = new Array(1 + 4).fill(0);
          for (let i = 0; i < count; ++i) {
            args[0] = base + i;
            const i3 = i * 3;
            let fx = (bytes[i3] / 255 - 0.5) * 2;
            let fy = (bytes[i3 + 1] / 255 - 0.5) * 2;
            const halfTheta = 0.5 * (bytes[i3 + 2] / 255) * Math.PI;

            const fz = 1 - (Math.abs(fx) + Math.abs(fy));
            const t = Math.max(-fz, 0);
            fx += fx >= 0 ? -t : t;
            fy += fy >= 0 ? -t : t;
            const axisLen = Math.sqrt(fx * fx + fy * fy + fz * fz);
            const zeroAxis = axisLen < 1e-6;
            const axisX = zeroAxis ? 1 : fx / axisLen;
            const axisY = zeroAxis ? 0 : fy / axisLen;
            const axisZ = zeroAxis ? 0 : fz / axisLen;
            const s = Math.sin(halfTheta);

            args[1] = axisX * s;
            args[2] = axisY * s;
            args[3] = axisZ * s;
            args[4] = Math.cos(halfTheta);
            callback.apply(null, args);
          }
          break;
        }
      }
    }
  }
}

function roundUp8(n: number): number {
  return (n + 7) & ~7;
}

export function unpackRad(
  fileBytes: Uint8Array,
  splatEncoding: SplatEncoding,
): {
  packedArray: Uint32Array | Uint32Array[];
  numSplats: number;
  extra: Record<string, unknown>;
} {
  const rad = new RadReader({ fileBytes });
  const numSplats = rad.numSplats;
  const maxSplats = computeMaxSplats(numSplats);
  const packedArray = splatEncoding.extended
    ? Array(2)
        .fill(null)
        .map(() => new Uint32Array(maxSplats * 4))
    : new Uint32Array(maxSplats * 4);
  const extra: Record<string, unknown> = {};

  const meta = rad.meta.lodMeta as RadLodMeta | undefined;
  if (meta) {
    extra.lodMeta = {
      pixelSizes: meta.pixelSizes,
      boundCenter: new THREE.Vector3(...meta.boundCenter),
      boundRadius: meta.boundRadius,
    };
  }

  rad.parseSplats({
    setEncoding: (encoding) => {
      Object.assign(splatEncoding, encoding);
    },
    center: (index, x, y, z) => {
      setPackedSplatCenter(packedArray, index, x, y, z);
    },
    alpha: (index, alpha) => {
      setPackedSplatOpacity(packedArray, index, alpha);
    },
    rgb: (index, r, g, b) => {
      setPackedSplatRgb(packedArray, index, r, g, b, splatEncoding);
    },
    scales: (index, scaleX, scaleY, scaleZ) => {
      setPackedSplatScales(
        packedArray,
        index,
        scaleX,
        scaleY,
        scaleZ,
        splatEncoding,
      );
    },
    quat: (index, quatX, quatY, quatZ, quatW) => {
      setPackedSplatQuat(packedArray, index, quatX, quatY, quatZ, quatW);
    },
    lodScales: (index, lodMin, lodLow, lodHigh, lodMax) => {
      setPackedSplatLods(
        packedArray,
        index,
        lodMin,
        lodLow,
        lodHigh,
        lodMax,
        splatEncoding,
      );
    },
    sh: (index, sh1, sh2, sh3) => {
      if (sh1) {
        if (!extra.sh1) {
          extra.sh1 = new Uint32Array(numSplats * 2);
        }
        encodeSh1Rgb(extra.sh1 as Uint32Array, index, sh1, splatEncoding);
      }
      if (sh2) {
        if (!extra.sh2) {
          extra.sh2 = new Uint32Array(numSplats * 4);
        }
        encodeSh2Rgb(extra.sh2 as Uint32Array, index, sh2, splatEncoding);
      }
      if (sh3) {
        if (!extra.sh3) {
          extra.sh3 = new Uint32Array(numSplats * 4);
        }
        encodeSh3Rgb(extra.sh3 as Uint32Array, index, sh3, splatEncoding);
      }
    },
  });
  return { packedArray, numSplats, extra };
}
