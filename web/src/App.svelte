<script lang="ts">
  import { onMount } from 'svelte';
  import Controls from './lib/Controls.svelte';
  import Gallery from './lib/Gallery.svelte';
  import History from './lib/History.svelte';
  import PaintCanvas from './lib/PaintCanvas.svelte';
  import { Editor } from './lib/state.svelte';

  const editor = new Editor();
  onMount(() => editor.init());

  function key(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !(e.target instanceof HTMLSelectElement)) {
      e.preventDefault();
      editor.undo();
    }
  }
</script>

<svelte:window onkeydown={key} />

<main>
  <header>
    <h1>GANPaint</h1>
    <p>Paint trees, grass, doors, sky, clouds, brick and domes into a GAN-generated church.</p>
  </header>

  {#if editor.error}
    <p class="error" role="alert">{editor.error}</p>
  {/if}

  {#if editor.info}
    <div class="layout">
      <PaintCanvas {editor} />
      <div class="side">
        <Controls {editor} />
        <Gallery {editor} />
        <History {editor} />
      </div>
    </div>
  {:else if !editor.error}
    <p>Loading…</p>
  {/if}
</main>

<style>
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 16px;
  }
  header h1 {
    margin: 0;
  }
  header p {
    margin: 4px 0 16px;
    color: var(--muted);
  }
  .layout {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }
  .side {
    display: flex;
    flex-direction: column;
    gap: 16px;
    flex: 1 1 320px;
    min-width: 0;
  }
  .error {
    padding: 8px 12px;
    border-radius: 6px;
    background: #d03c4033;
    border: 1px solid #d03c40;
  }
</style>
