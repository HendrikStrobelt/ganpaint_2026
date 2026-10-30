<script lang="ts">
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();
</script>

<section class="controls" aria-label="Painting tools">
  <div class="group" role="radiogroup" aria-label="Action">
    <button role="radio" aria-checked={editor.mode === 'draw'} class="draw" onclick={() => (editor.mode = 'draw')}>
      Draw
    </button>
    <button role="radio" aria-checked={editor.mode === 'remove'} class="remove" onclick={() => (editor.mode = 'remove')}>
      Remove
    </button>
  </div>

  <div class="group brushes" role="radiogroup" aria-label="Brush">
    {#each editor.info?.brushes ?? [] as b (b.name)}
      <button role="radio" aria-checked={editor.brush === b.name} onclick={() => (editor.brush = b.name)}>
        {b.label}
      </button>
    {/each}
  </div>

  <label class="strength">
    Strength
    <select bind:value={editor.strength}>
      {#each Object.entries(editor.info?.strengths ?? {}) as [name, units] (name)}
        <option value={name}>{name} ({units} units)</option>
      {/each}
    </select>
  </label>

  <div class="group actions">
    <button onclick={() => editor.undo()} disabled={!editor.strokes.length}>Undo</button>
    <button onclick={() => editor.reset()} disabled={!editor.strokes.length}>Reset</button>
    <a class="button" href={editor.image} download={`ganpaint-${editor.sampleId}.${editor.image.startsWith('data:image/webp') || editor.image.endsWith('.webp') ? 'webp' : 'png'}`}>
      Download
    </a>
  </div>
</section>

<style>
  .controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: min(100%, 512px);
  }
  .group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .strength {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
  }
  button[aria-checked='true'] {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  button.draw[aria-checked='true'] {
    background: #0aa5b1;
    border-color: #0aa5b1;
  }
  button.remove[aria-checked='true'] {
    background: #d03c40;
    border-color: #d03c40;
  }
</style>
