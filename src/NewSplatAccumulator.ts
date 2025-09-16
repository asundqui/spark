import * as THREE from "three";
import { dyno } from ".";
import { PackedSplats, type SplatEncoding } from "./PackedSplats";
import type { GeneratorMapping, GeneratorState } from "./SplatAccumulator";
import { SplatGenerator, SplatTransformer } from "./SplatGenerator";
import { SplatMesh } from "./SplatMesh";

export class NewSplatAccumulator {
  splats: PackedSplats;
  originToWorld = new THREE.Matrix4();
  time = 0;
  mapping: Map<SplatGenerator, GeneratorMapping> = new Map();

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
  }: {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    time?: number;
    sortMapping: Map<SplatGenerator, GeneratorMapping>;
    lastSplats?: NewSplatAccumulator;
    origin?: THREE.Vector3;
    originToWorld?: THREE.Matrix4;
    camera?: THREE.Camera;
    renderSize?: THREE.Vector2;
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
    const allNewState = new Map<SplatGenerator, GeneratorState>();

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
          globalEdits: [],
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

    const splatCounts = visibleGenerators.map((g) => g.numSplats);
    const { maxSplats, mapping } = this.splats.generateMapping(splatCounts);
    this.ensureGenerate(maxSplats);
    let totalSplats = 0;

    this.mapping = mapping.reduce((map, { base, count }, index) => {
      const node = visibleGenerators[index];
      const state = allNewState.get(node);

      const generator = node.generator;
      const version = renderer.info.render.frame;
      if (generator && count > 0) {
        try {
          this.splats.generate({
            renderer,
            generator,
            base,
            count,
          });
        } catch (error) {
          node.generator = undefined;
          node.generatorError = error;
        }
      }
      map.set(node, { node, generator, version, base, count, state });
      totalSplats = Math.max(totalSplats, base + count);
      return map;
    }, new Map<SplatGenerator, GeneratorMapping>());

    this.splats.numSplats = totalSplats;
  }
}
