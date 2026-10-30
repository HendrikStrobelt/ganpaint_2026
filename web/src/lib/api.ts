export type Mode = 'draw' | 'remove';

export interface Brush {
  name: string;
  label: string;
}

export interface ProjectInfo {
  name: string;
  layer: string;
  canvas_grid: number;
  strengths: Record<string, number>;
  default_strength: string;
  brushes: Brush[];
  gallery: { id: number; thumb: string }[];
  max_id: number;
}

export interface StrokePayload {
  brush: string;
  mode: Mode;
  strength: string;
  mask: string; // PNG data URL on a canvas_grid x canvas_grid canvas
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function listProjects(): Promise<{ name: string; layer: string }[]> {
  return json(await fetch('/api/projects'));
}

export async function getProject(name: string): Promise<ProjectInfo> {
  return json(await fetch(`/api/projects/${encodeURIComponent(name)}`));
}

export const sampleUrl = (project: string, id: number) =>
  `/api/projects/${encodeURIComponent(project)}/samples/${id}.webp`;

export async function generate(
  project: string,
  id: number,
  strokes: StrokePayload[],
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`/api/projects/${encodeURIComponent(project)}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, strokes }),
    signal,
  });
  return (await json<{ image: string }>(res)).image;
}
