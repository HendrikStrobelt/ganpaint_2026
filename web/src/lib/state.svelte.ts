import { generate, getProject, listProjects, sampleUrl, type Mode, type ProjectInfo, type StrokePayload } from './api';

export interface Stroke extends StrokePayload {
  key: number;
}

export class Editor {
  info = $state<ProjectInfo | null>(null);
  sampleId = $state(0);
  strokes = $state<Stroke[]>([]);
  brush = $state('');
  mode = $state<Mode>('draw');
  strength = $state('high');
  image = $state('');
  busy = $state(false);
  error = $state('');
  hovered = $state<number | null>(null);

  #nextKey = 1;
  #request = 0;
  #abort: AbortController | null = null;

  async init() {
    try {
      const projects = await listProjects();
      if (!projects.length) throw new Error('the server has no projects');
      const info = await getProject(projects[0].name);
      this.info = info;
      this.brush = info.brushes[0]?.name ?? '';
      this.strength = info.default_strength;
      const ids = info.gallery.map((g) => g.id);
      await this.select(ids[Math.floor(Math.random() * ids.length)] ?? 0);
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    }
  }

  async select(id: number) {
    this.sampleId = id;
    this.strokes = [];
    await this.refresh();
  }

  async addStroke(mask: string) {
    this.strokes.push({ key: this.#nextKey++, brush: this.brush, mode: this.mode, strength: this.strength, mask });
    await this.refresh();
  }

  async undo() {
    if (this.strokes.pop()) await this.refresh();
  }

  async reset() {
    if (this.strokes.length) {
      this.strokes = [];
      await this.refresh();
    }
  }

  async removeStroke(key: number) {
    this.strokes = this.strokes.filter((s) => s.key !== key);
    await this.refresh();
  }

  /** Re-render the current sample with all strokes; only the latest request may update the image. */
  async refresh() {
    if (!this.info) return;
    const request = ++this.#request;
    this.#abort?.abort();
    if (!this.strokes.length) {
      this.image = sampleUrl(this.info.name, this.sampleId);
      this.busy = false;
      this.error = '';
      return;
    }
    const abort = (this.#abort = new AbortController());
    this.busy = true;
    try {
      const image = await generate(
        this.info.name,
        this.sampleId,
        this.strokes.map(({ brush, mode, strength, mask }) => ({ brush, mode, strength, mask })),
        abort.signal,
      );
      if (request === this.#request) {
        this.image = image;
        this.error = '';
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError') && request === this.#request) {
        this.error = e instanceof Error ? e.message : String(e);
      }
    } finally {
      if (request === this.#request) this.busy = false;
    }
  }
}
