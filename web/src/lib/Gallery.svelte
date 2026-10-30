<script lang="ts">
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();
</script>

<section class="gallery" aria-label="Choose a church">
  {#each editor.info?.gallery ?? [] as g (g.id)}
    <button class="thumb" aria-pressed={editor.sampleId === g.id} onclick={() => editor.select(g.id)} aria-label={`Church ${g.id}`}>
      <img src={g.thumb} alt="" loading="lazy" />
    </button>
  {/each}
</section>

<style>
  .gallery {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 6px;
    width: min(100%, 512px);
  }
  .thumb {
    flex: 0 0 auto;
    padding: 0;
    width: 64px;
    height: 64px;
    border-width: 2px;
    overflow: hidden;
  }
  .thumb[aria-pressed='true'] {
    border-color: var(--accent);
  }
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
</style>
