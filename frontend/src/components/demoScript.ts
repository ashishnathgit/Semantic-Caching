// The stage-demo presets live in data/demo_script.json, shared with
// `python -m semcache preflight`, scripts/check_demo.py and scripts/screenshots.mjs.
import script from "../../../data/demo_script.json";

export interface DemoStep {
  label: string;
  query: string;
  expect: "hit" | "miss" | "follow-up" | "blocked";
  note: string;
  after?: number; // a preset that must have been answered first
}

export const DEMO_SCRIPT = script.steps as DemoStep[];
