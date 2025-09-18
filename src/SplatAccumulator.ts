import * as THREE from "three";

import { PackedSplats, type SplatEncoding } from "./PackedSplats";
import type {
  GsplatGenerator,
  SplatGenerator,
  SplatModifier,
} from "./SplatGenerator";

// SplatAccumulator helps manage the generation of splats from multiple
// SplatGenerators, keeping track of the splat mapping, coordinate system,
// and reference count.

export type GeneratorState = object;

// A GeneratorMapping describes a Gsplat range that was generated, including
// which generator and its version number.
export type GeneratorMapping = {
  object: SplatGenerator;
  multi?: unknown;
  generator?: GsplatGenerator;
  version: number;
  base: number;
  count: number;
  state?: GeneratorState;
};

export class SplatAccumulator {
  splats: PackedSplats;
  // The transform from Accumulator coordinate system to world coordinates.
  toWorld = new THREE.Matrix4();
  // An array of all Gsplat mappings that were used for generation
  mapping: GeneratorMapping[] = [];
  // Number of SparkViewpoints (or other) that reference this accumulator, used
  // to figure out when it can be recycled for use
  refCount = 0;

  // Incremented every time the splats are updated/generated.
  splatsVersion = -1;
  // Incremented every time the splat mapping/layout is updated.
  // Splat sort order can be reused between equivalent mapping versions.
  mappingVersion = -1;

  constructor({ splatEncoding }: { splatEncoding: SplatEncoding }) {
    this.splats = new PackedSplats({ splatEncoding });
  }

  ensureGenerate(maxSplats: number) {
    if (this.splats.ensureGenerate(maxSplats)) {
      // If we had to resize our PackedSplats then clear all previous mappings
      this.mapping = [];
    }
  }

  // Generate all Gsplats from an array of generators
  generateSplats({
    renderer,
    modifier,
    generators,
    forceUpdate,
    originToWorld,
  }: {
    renderer: THREE.WebGLRenderer;
    modifier: SplatModifier;
    generators: GeneratorMapping[];
    forceUpdate?: boolean;
    originToWorld: THREE.Matrix4;
  }) {
    // Create a lookup from last SplatGenerator
    const mapping = this.mapping.reduce((map, record) => {
      map.set(record.object, record);
      return map;
    }, new Map<SplatGenerator, GeneratorMapping>());

    // Run generators that are different from existing mapping
    let updated = 0;
    let numSplats = 0;
    for (const { object, generator, version, base, count } of generators) {
      const current = mapping.get(object);
      if (
        forceUpdate ||
        generator !== current?.generator ||
        version !== current?.version ||
        base !== current?.base ||
        count !== current?.count
      ) {
        // Something is different from before so we should generate these Gsplats
        if (generator && count > 0) {
          const modGenerator = modifier.apply(generator);
          try {
            this.splats.generate({
              generator: modGenerator,
              base,
              count,
              renderer,
            });
          } catch (error) {
            object.generator = undefined;
            object.generatorError = error;
          }
          updated += 1;
        }
      }
      numSplats = Math.max(numSplats, base + count);
    }

    this.splats.numSplats = numSplats;
    this.toWorld = originToWorld;
    this.mapping = generators;
    return updated !== 0;
  }

  // Check if this accumulator has exactly the same generator mapping as
  // the previous one. If so, we can reuse the Gsplat sort order.
  hasCorrespondence(other: SplatAccumulator) {
    if (this.mapping.length !== other.mapping.length) {
      return false;
    }
    return this.mapping.every(({ object, base, count }, i) => {
      const {
        object: otherObject,
        base: otherBase,
        count: otherCount,
      } = other.mapping[i];
      return (
        object === otherObject && base === otherBase && count === otherCount
      );
    });
  }
}
