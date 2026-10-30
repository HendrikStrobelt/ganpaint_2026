<script lang="ts">
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();

  const COLORS = { draw: '#08bdba', remove: '#fa4d56' } as const;
  const RADIUS = 1; // brush radius in canvas-grid cells, as in the original UI

  let overlay: HTMLCanvasElement;
  let mask: HTMLCanvasElement | null = null;
  let last: [number, number] | null = null;
  let grid = $derived(editor.info?.canvas_grid ?? 32);

  function cell(e: PointerEvent): [number, number] {
    const r = overlay.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * grid, ((e.clientY - r.top) / r.height) * grid];
  }

  function stamp([x, y]: [number, number]) {
    const m = mask!.getContext('2d')!;
    m.fillStyle = '#000';
    m.beginPath();
    m.ellipse(x, y, RADIUS, RADIUS, 0, 0, 2 * Math.PI);
    m.fill();
    const o = overlay.getContext('2d')!;
    const s = overlay.width / grid;
    o.globalAlpha = 0.7;
    o.fillStyle = COLORS[editor.mode];
    o.beginPath();
    o.ellipse(x * s, y * s, RADIUS * s, RADIUS * s, 0, 0, 2 * Math.PI);
    o.fill();
  }

  function down(e: PointerEvent) {
    if (!editor.info || e.button !== 0) return;
    overlay.setPointerCapture(e.pointerId);
    mask = document.createElement('canvas');
    mask.width = mask.height = grid;
    last = cell(e);
    stamp(last);
  }

  function move(e: PointerEvent) {
    if (!mask || !last) return;
    const p = cell(e);
    const steps = Math.max(1, Math.ceil(Math.hypot(p[0] - last[0], p[1] - last[1]) / 0.5));
    for (let i = 1; i <= steps; i++) {
      stamp([last[0] + ((p[0] - last[0]) * i) / steps, last[1] + ((p[1] - last[1]) * i) / steps]);
    }
    last = p;
  }

  // Hovering a stroke in the history highlights where it was painted.
  $effect(() => {
    const stroke = editor.strokes.find((s) => s.key === editor.hovered);
    if (mask || !overlay) return;
    const o = overlay.getContext('2d')!;
    o.clearRect(0, 0, overlay.width, overlay.height);
    if (!stroke) return;
    const img = new Image();
    img.onload = () => {
      if (mask || editor.hovered !== stroke.key) return;
      const tint = document.createElement('canvas');
      tint.width = tint.height = overlay.width;
      const t = tint.getContext('2d')!;
      t.imageSmoothingEnabled = false;
      t.drawImage(img, 0, 0, tint.width, tint.height);
      t.globalCompositeOperation = 'source-in';
      t.fillStyle = COLORS[stroke.mode];
      t.fillRect(0, 0, tint.width, tint.height);
      o.globalAlpha = 0.7;
      o.drawImage(tint, 0, 0);
    };
    img.src = stroke.mask;
  });

  function up() {
    if (!mask) return;
    const png = mask.toDataURL('image/png');
    mask = last = null;
    overlay.getContext('2d')!.clearRect(0, 0, overlay.width, overlay.height);
    editor.addStroke(png);
  }
</script>

<div class="stage" class:busy={editor.busy}>
  <img src={editor.image} alt="Generated church. Paint on it to add or remove features." draggable="false" />
  <canvas
    bind:this={overlay}
    width="512"
    height="512"
    aria-label="Painting surface"
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
  ></canvas>
  {#if editor.busy}<span class="spinner" role="status" aria-label="Generating"></span>{/if}
</div>

<style>
  .stage {
    position: relative;
    width: min(100%, 512px);
    aspect-ratio: 1;
    overflow: hidden;
    background: var(--cds-layer-01, #262626);
  }
  img,
  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  img {
    object-fit: cover;
    user-select: none;
    transition: opacity 0.15s;
  }
  .busy img {
    opacity: 0.8;
  }
  canvas {
    touch-action: none;
    cursor: crosshair;
  }
  .spinner {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 20px;
    height: 20px;
    border: 3px solid rgb(255 255 255 / 0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
