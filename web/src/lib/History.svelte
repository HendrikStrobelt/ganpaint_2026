<script lang="ts">
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();
</script>

{#if editor.strokes.length}
  <section class="history" aria-label="Strokes">
    <h2>Strokes</h2>
    <ol>
      {#each editor.strokes as s, i (s.key)}
        <li onmouseenter={() => (editor.hovered = s.key)} onmouseleave={() => (editor.hovered = null)}>
          <span class="dot {s.mode}" aria-hidden="true"></span>
          <span class="name">{i + 1}. {s.mode} {s.brush} ({s.strength})</span>
          <button class="small" aria-label={`Delete stroke ${i + 1}`} onclick={() => editor.removeStroke(s.key)}>✕</button>
        </li>
      {/each}
    </ol>
  </section>
{/if}

<style>
  .history {
    width: min(100%, 512px);
  }
  h2 {
    margin: 0 0 6px;
    font-size: 0.95rem;
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 180px;
    overflow-y: auto;
  }
  li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 6px;
    border-radius: 6px;
    background: var(--surface-2);
    font-size: 0.9rem;
  }
  li:hover {
    outline: 1px solid var(--accent);
  }
  .name {
    flex: 1;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }
  .dot.draw {
    background: #0bc6d4;
  }
  .dot.remove {
    background: #e1484a;
  }
  .small {
    padding: 0 6px;
    min-height: 24px;
  }
</style>
