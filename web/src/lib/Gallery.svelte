<script lang="ts">
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();
</script>

<section class="gallery" aria-label="Choose a church">
  <h2>Churches</h2>
  <div class="strip">
    {#each editor.info?.gallery ?? [] as g (g.id)}
      <button class="thumb" aria-pressed={editor.sampleId === g.id} onclick={() => editor.select(g.id)} aria-label={`Church ${g.id}`}>
        <img src={g.thumb} alt="" loading="lazy" />
      </button>
    {/each}
  </div>
</section>

<style>
  .gallery {
    width: min(100%, 512px);
  }
  h2 {
    margin: 0 0 8px;
    font-size: 0.875rem;
    font-weight: 400;
    color: var(--muted);
  }
  .strip {
    display: flex;
    gap: 4px;
    overflow-x: auto;
    padding-bottom: 6px;
  }
  .thumb {
    flex: 0 0 auto;
    padding: 0;
    width: 64px;
    height: 64px;
    border: 2px solid transparent;
    border-radius: 0;
    background: none;
    cursor: pointer;
    overflow: hidden;
  }
  .thumb:hover {
    border-color: var(--cds-border-strong-01, #6f6f6f);
  }
  .thumb[aria-pressed='true'] {
    border-color: var(--cds-focus, #fff);
  }
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
</style>
