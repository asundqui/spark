import * as THREE from "three";
import { SplatEdit, dyno } from ".";
import { PackedSplats, type SplatEncoding } from "./PackedSplats";
import type { GeneratorMapping, GeneratorState } from "./SplatAccumulator";
import { SplatGenerator, SplatTransformer } from "./SplatGenerator";
import { SplatMesh } from "./SplatMesh";

export class NewSplatAccumulator {
  splats: PackedSplats;
  originToWorld = new THREE.Matrix4();
  time = 0;
  mapping: Map<unknown, GeneratorMapping> = new Map();

  worldToOrigin = new SplatTransformer();
  viewToWorld = new THREE.Matrix4();

  modifier = dyno.dynoBlock(
    { gsplat: dyno.Gsplat },
    { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      if (!gsplat) {
        throw new Error("gsplat not defined");
      }
      gsplat = this.worldToOrigin.applyGsplat(gsplat);
      return { gsplat };
    },
  );

  constructor({ splatEncoding }: { splatEncoding: SplatEncoding }) {
    this.splats = new PackedSplats({ splatEncoding });
  }

  ensureGenerate(maxSplats: number) {
    if (this.splats.ensureGenerate(maxSplats)) {
      // this.mapping.clear();
    }
  }

  generateSplats({
    renderer,
    scene,
    time,
    sortMapping,
    lastSplats,
    origin,
    originToWorld,
    camera,
    renderSize,
    globalLodScale,
  }: {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    time?: number;
    sortMapping: Map<unknown, GeneratorMapping>;
    lastSplats?: NewSplatAccumulator;
    origin?: THREE.Vector3;
    originToWorld?: THREE.Matrix4;
    camera?: THREE.Camera;
    renderSize?: THREE.Vector2;
    globalLodScale: number;
  }) {
    if (origin) {
      this.originToWorld.makeTranslation(origin.x, origin.y, origin.z);
    }
    if (originToWorld) {
      this.originToWorld.copy(originToWorld);
    }

    this.worldToOrigin.updateFromMatrix(this.originToWorld.clone().invert());

    let layers: THREE.Layers | null = null;
    if (camera) {
      camera.updateMatrixWorld();
      this.viewToWorld.copy(camera.matrixWorld);
      layers = camera.layers;
    }

    this.time = time ?? 0;
    const deltaTime = time ? time - (lastSplats?.time ?? time) : 0;

    const allGenerators: SplatGenerator[] = [];
    scene.traverse((node) => {
      if (node instanceof SplatGenerator) {
        if (!layers || layers.test(node.layers)) {
          allGenerators.push(node);
        }
      }
    });
    const allNewState = new Map<unknown, GeneratorState>();

    const globalEditsSet = new Set<SplatEdit>();
    scene.traverseVisible((node) => {
      if (node instanceof SplatEdit) {
        let ancestor = node.parent;
        while (ancestor != null && !(ancestor instanceof SplatMesh)) {
          ancestor = ancestor.parent;
        }
        if (ancestor == null) {
          // Not part of a SplatMesh so it's a global edit
          globalEditsSet.add(node);
        }
      }
    });
    const globalEdits = Array.from(globalEditsSet);

    for (const object of allGenerators) {
      const sortState = sortMapping.get(object)?.state;
      const lastState = lastSplats?.mapping.get(object)?.state;
      try {
        const newState = {};
        object.frameUpdate?.({
          object,
          time: this.time,
          deltaTime,
          viewToWorld: this.viewToWorld,
          camera,
          renderSize,
          globalEdits,
          globalLodScale,
          sortState,
          lastState,
          newState,
        });
        allNewState.set(object, newState);
      } catch (error) {
        object.generator = undefined;
        object.generatorError = error;
      }
    }

    const visibleGenerators: SplatGenerator[] = [];
    scene.traverseVisible((node) => {
      if (node instanceof SplatGenerator) {
        if (!layers || layers.test(node.layers)) {
          visibleGenerators.push(node);
        }
      }
    });

    const splatChunks: {
      object: SplatGenerator;
      multi?: unknown;
      numSplats: number;
    }[] = [];
    for (const generator of visibleGenerators) {
      splatChunks.push({ object: generator, numSplats: generator.numSplats });
      if (generator.multiSplats) {
        for (const [multi, count] of generator.multiSplats.entries()) {
          splatChunks.push({ object: generator, multi, numSplats: count });
        }
      }
    }

    const splatCounts = splatChunks.map((g) => g.numSplats);
    const { maxSplats, mapping } = this.splats.generateMapping(splatCounts);
    this.ensureGenerate(maxSplats);
    let totalSplats = 0;

    this.mapping = mapping.reduce((map, { base, count }, index) => {
      const { object, multi } = splatChunks[index];
      const reference = multi ?? object;
      const state = allNewState.get(reference) ?? {};

      const generator = object.generator;
      const version = renderer.info.render.frame;
      if (generator && count > 0) {
        try {
          if (multi) {
            const sortState = sortMapping.get(multi)?.state;
            const lastState = lastSplats?.mapping.get(multi)?.state;
            object.prepareMulti?.({
              object,
              multi,
              sortState,
              lastState,
              newState: state,
            });
          }
          this.splats.generate({
            renderer,
            generator,
            base,
            count,
          });
        } catch (error) {
          object.generator = undefined;
          object.generatorError = error;
        }
      }
      map.set(reference, {
        object,
        multi,
        generator,
        version,
        base,
        count,
        state,
      });
      totalSplats = Math.max(totalSplats, base + count);
      return map;
    }, new Map<unknown, GeneratorMapping>());

    this.splats.numSplats = totalSplats;
  }
}
